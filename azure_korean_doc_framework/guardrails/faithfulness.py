import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.multi_model_manager import MultiModelManager


@dataclass
class FaithfulnessResult:
    faithfulness_score: float = 1.0
    distortions: List[str] = field(default_factory=list)
    verdict: str = "FAITHFUL"


class FaithfulnessChecker:
    """Validate that an answer is grounded in retrieved documents.

    v4.6 behavior intentionally ignores appended citation lines and short-circuits
    simple extraction answers when the exact answer text appears verbatim in the
    source documents.
    """

    def __init__(self, model_manager: MultiModelManager, threshold: float = 0.85):
        self.model_manager = model_manager
        self.threshold = threshold

    def verify(self, answer: str, documents: List[str], model_key: Optional[str] = None) -> FaithfulnessResult:
        normalized_answer = self._normalize_for_check(answer)
        if not normalized_answer:
            return FaithfulnessResult(faithfulness_score=1.0, distortions=[], verdict="FAITHFUL")

        heuristic_result = self._heuristic_short_answer_check(normalized_answer, documents)
        if heuristic_result is not None:
            return heuristic_result

        docs_text = "\n---\n".join(documents)
        prompt = (
            "다음 문서와 생성된 답변을 비교해 답변이 원문을 왜곡했는지 검증하세요.\n"
            "특히 숫자, 기간, 횟수, 고유명사를 확인하세요.\n"
            "출처 표기([출처: ...])는 평가 대상이 아니므로 무시하세요.\n"
            "답변이 짧은 엔티티/값 추출 결과라면, 문서에 동일 표현이 존재하는지 우선 보세요.\n"
            "출력 형식:\n"
            f"faithfulness_score: 0.0 ~ 1.0\n"
            "distortions: [왜곡 항목]\n"
            f"verdict: FAITHFUL 또는 UNFAITHFUL (threshold={self.threshold})\n\n"
            f"[문서]\n{docs_text}\n\n[답변]\n{normalized_answer}"
        )
        response = self.model_manager.get_completion(
            prompt=prompt,
            model_key=model_key,
            system_message="당신은 답변 충실도 검증기입니다.",
            temperature=0.0,
            max_tokens=300,
        )
        return self._parse_result(response)

    def _parse_result(self, response: str) -> FaithfulnessResult:
        result = FaithfulnessResult()
        collect_distortions = False
        for line in (response or "").splitlines():
            line = line.strip()
            if line.lower().startswith("faithfulness_score"):
                match = re.search(r"[\d.]+", line)
                if match:
                    result.faithfulness_score = float(match.group())
            elif line.lower().startswith("distortions"):
                claims = line.split(":", 1)[-1].strip().strip("[]")
                if claims:
                    result.distortions = [item.strip().strip("'\"") for item in claims.split(",") if item.strip()]
                collect_distortions = True
            elif line.lower().startswith("verdict"):
                result.verdict = line.split(":", 1)[-1].strip().upper()
                collect_distortions = False
            elif collect_distortions and line:
                cleaned = line.strip("- •\t ").strip("'\"")
                if cleaned:
                    result.distortions.append(cleaned)

        result.verdict = "FAITHFUL" if result.faithfulness_score >= self.threshold else "UNFAITHFUL"
        result.distortions = list(dict.fromkeys(result.distortions))
        return result

    def _normalize_for_check(self, answer: str) -> str:
        """Remove empty lines and `[출처: ...]` suffixes before model-based evaluation."""
        lines = []
        for raw_line in (answer or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[출처:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _heuristic_short_answer_check(self, normalized_answer: str, documents: List[str]) -> Optional[FaithfulnessResult]:
        """Fast-path short extraction answers when the same surface form exists in the source text."""
        compact_answer = re.sub(r"\s+", " ", normalized_answer).strip()
        if not compact_answer:
            return FaithfulnessResult(faithfulness_score=1.0, distortions=[], verdict="FAITHFUL")
        if len(compact_answer) > 40 or "\n" in normalized_answer:
            return None

        normalized_docs = [re.sub(r"\s+", " ", doc or " ").strip() for doc in documents]
        if any(compact_answer in doc for doc in normalized_docs):
            return FaithfulnessResult(faithfulness_score=1.0, distortions=[], verdict="FAITHFUL")
        return None
