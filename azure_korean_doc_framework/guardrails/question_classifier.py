"""질문 유형 분류 모듈.

사용자 질문을 3가지 유형으로 분류하여 최적의 처리 경로를 결정합니다:
  - regulatory: 숫자/기간/횟수 관련 규정형 질문
  - extraction: 이름/목록 등 직접 추출형 질문
  - explanatory: 설명형 질문 (기본값)

Agent Routing과 Evidence Extraction 전략 선택에 사용됩니다.
"""

import re
from dataclasses import dataclass


@dataclass
class QuestionType:
    """질문 분류 결과.

    Attributes:
        category: 질문 유형 ('regulatory', 'extraction', 'explanatory').
        reason: 분류 사유.
    """
    category: str
    reason: str


class QuestionClassifier:
    """한국어 정규식 기반 질문 유형 분류기."""
    EXTRACTION_PATTERNS = [r"무엇", r"이름", r"명칭", r"목록", r"어디", r"누구"]
    REGULATORY_PATTERNS = [r"몇", r"횟수", r"주기", r"기간", r"%", r"원", r"회", r"기준"]

    def classify(self, question: str) -> QuestionType:
        if any(re.search(pattern, question) for pattern in self.REGULATORY_PATTERNS):
            return QuestionType(category="regulatory", reason="numeric_or_policy")
        if any(re.search(pattern, question) for pattern in self.EXTRACTION_PATTERNS):
            return QuestionType(category="extraction", reason="direct_lookup")
        return QuestionType(category="explanatory", reason="default")
