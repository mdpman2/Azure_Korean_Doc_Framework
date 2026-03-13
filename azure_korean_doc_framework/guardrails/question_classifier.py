import re
from dataclasses import dataclass


@dataclass
class QuestionType:
    category: str
    reason: str


class QuestionClassifier:
    EXTRACTION_PATTERNS = [r"무엇", r"이름", r"명칭", r"목록", r"어디", r"누구"]
    REGULATORY_PATTERNS = [r"몇", r"횟수", r"주기", r"기간", r"%", r"원", r"회", r"기준"]

    def classify(self, question: str) -> QuestionType:
        if any(re.search(pattern, question) for pattern in self.REGULATORY_PATTERNS):
            return QuestionType(category="regulatory", reason="numeric_or_policy")
        if any(re.search(pattern, question) for pattern in self.EXTRACTION_PATTERNS):
            return QuestionType(category="extraction", reason="direct_lookup")
        return QuestionType(category="explanatory", reason="default")
