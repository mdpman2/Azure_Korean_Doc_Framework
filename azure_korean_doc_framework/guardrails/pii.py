import re
from dataclasses import dataclass
from typing import List


@dataclass
class PIIMatch:
    match_type: str
    text: str


class KoreanPIIDetector:
    def __init__(self):
        self.patterns = {
            "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            "phone": re.compile(r"01[0-9]-?\d{3,4}-?\d{4}"),
            "resident_id": re.compile(r"\d{6}-?[1-4]\d{6}"),
            "credit_card": re.compile(r"(?:\d{4}-){3}\d{4}"),
        }

    def detect(self, text: str) -> List[PIIMatch]:
        matches: List[PIIMatch] = []
        for match_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(match_type=match_type, text=match.group()))
        return matches

    def mask(self, text: str) -> str:
        masked = text
        for pii_match in self.detect(text):
            masked = masked.replace(pii_match.text, self._mask_value(pii_match.text))
        return masked

    def _mask_value(self, value: str) -> str:
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
