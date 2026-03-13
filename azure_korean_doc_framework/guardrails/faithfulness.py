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
    def __init__(self, model_manager: MultiModelManager, threshold: float = 0.85):
        self.model_manager = model_manager
        self.threshold = threshold

    def verify(self, answer: str, documents: List[str], model_key: Optional[str] = None) -> FaithfulnessResult:
        docs_text = "\n---\n".join(documents)
        prompt = (
            "다음 문서와 생성된 답변을 비교해 답변이 원문을 왜곡했는지 검증하세요.\n"
            "특히 숫자, 기간, 횟수, 고유명사를 확인하세요.\n"
            "출력 형식:\n"
            f"faithfulness_score: 0.0 ~ 1.0\n"
            "distortions: [왜곡 항목]\n"
            f"verdict: FAITHFUL 또는 UNFAITHFUL (threshold={self.threshold})\n\n"
            f"[문서]\n{docs_text}\n\n[답변]\n{answer}"
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
            elif line.lower().startswith("verdict"):
                result.verdict = line.split(":", 1)[-1].strip().upper()

        result.verdict = "FAITHFUL" if result.faithfulness_score >= self.threshold else "UNFAITHFUL"
        return result
