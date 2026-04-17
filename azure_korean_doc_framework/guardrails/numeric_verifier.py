"""숫자/금액/기간 검증 모듈.

생성된 답변에 포함된 숫자, 금액, 기간, 횟수 등이
실제 검색 문맥(context)에 존재하는지 검증합니다.
한국어 숫자 단위(%, 원, 명, 개월 등)와 동의 표현(반기↔6개월 등)을
인식하여 정확한 근거 매칭을 수행합니다.

Usage:
    verifier = NumericVerifier()
    result = verifier.verify(answer_text, context_texts)
    if not result.passed:
        print(f"근거 없는 숫자: {result.ungrounded_numbers}")
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class NumericVerification:
    """숫자 검증 결과.

    Attributes:
        passed: 모든 숫자가 문맥에 근거하면 True.
        ungrounded_numbers: 문맥에서 찾을 수 없는 숫자 표현 목록.
        total_numbers_found: 답변에서 추출된 총 숫자 수.
    """
    passed: bool
    ungrounded_numbers: List[str] = field(default_factory=list)
    total_numbers_found: int = 0


class NumericVerifier:
    """한국어 숫자 표현 근거 검증기.

    정규식으로 답변에서 숫자 표현을 추출하고, 동의 표현 정규화 후
    검색 문맥에 실제 존재하는지 확인합니다.
    """

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
