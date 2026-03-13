import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class NumericVerification:
    passed: bool
    ungrounded_numbers: List[str] = field(default_factory=list)
    total_numbers_found: int = 0


class NumericVerifier:
    NUMERIC_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|회|번|개|건|명|원|만원|억원|조|개월|월|년|일|시간|분기)?")

    def __init__(self):
        self.equivalence_pairs = {
            "반기": "6개월",
            "분기": "3개월",
            "1년": "12개월",
            "연 1회": "1년 1회",
        }

    def verify(self, answer: str, context_texts: List[str]) -> NumericVerification:
        answer_numbers = self._extract_numbers(answer)
        if not answer_numbers:
            return NumericVerification(passed=True, total_numbers_found=0)

        context_joined = self._normalize(" ".join(context_texts))
        ungrounded = []
        for number in answer_numbers:
            if not self._is_grounded(number, context_joined):
                ungrounded.append(number)

        return NumericVerification(
            passed=len(ungrounded) == 0,
            ungrounded_numbers=ungrounded,
            total_numbers_found=len(answer_numbers),
        )

    def _extract_numbers(self, text: str) -> List[str]:
        return [match.group().strip() for match in re.finditer(self.NUMERIC_PATTERN, text)]

    def _normalize(self, text: str) -> str:
        normalized = text.replace(",", "")
        for left, right in self.equivalence_pairs.items():
            normalized = normalized.replace(left, right)
        return normalized

    def _is_grounded(self, number_expression: str, context_normalized: str) -> bool:
        candidate = self._normalize(number_expression)
        return candidate in context_normalized
