import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.multi_model_manager import MultiModelManager


@dataclass
class HallucinationResult:
    grounded_ratio: float = 1.0
    ungrounded_claims: List[str] = field(default_factory=list)
    verdict: str = "PASS"


class HallucinationDetector:
    def __init__(self, model_manager: MultiModelManager, threshold: float = 0.8):
        self.model_manager = model_manager
        self.threshold = threshold

    def verify(self, answer: str, documents: List[str], model_key: Optional[str] = None) -> HallucinationResult:
        docs_text = "\n---\n".join(documents)
        prompt = (
            "다음 검색 문서와 답변을 비교해 문서에 근거하지 않은 주장이 있는지 판정하세요.\n"
            "출력 형식:\n"
            "grounded_ratio: 0.0 ~ 1.0\n"
            "ungrounded_claims: [근거 없는 주장]\n"
            f"verdict: PASS 또는 FAIL (threshold={self.threshold})\n\n"
            f"[문서]\n{docs_text}\n\n[답변]\n{answer}"
        )
        response = self.model_manager.get_completion(
            prompt=prompt,
            model_key=model_key,
            system_message="당신은 할루시네이션 검증기입니다.",
            temperature=0.0,
            max_tokens=300,
        )
        return self._parse_result(response)

    def _parse_result(self, response: str) -> HallucinationResult:
        result = HallucinationResult()
        for line in (response or "").splitlines():
            line = line.strip()
            if line.lower().startswith("grounded_ratio"):
                match = re.search(r"[\d.]+", line)
                if match:
                    result.grounded_ratio = float(match.group())
            elif line.lower().startswith("ungrounded_claims"):
                claims = line.split(":", 1)[-1].strip().strip("[]")
                if claims:
                    result.ungrounded_claims = [item.strip().strip("'\"") for item in claims.split(",") if item.strip()]
            elif line.lower().startswith("verdict"):
                result.verdict = line.split(":", 1)[-1].strip().upper()

        result.verdict = "PASS" if result.grounded_ratio >= self.threshold else "FAIL"
        return result
