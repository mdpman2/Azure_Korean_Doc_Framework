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
        seen_spans: set = set()
        for match_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    matches.append(PIIMatch(match_type=match_type, text=match.group()))
        return matches

    def mask(self, text: str) -> str:
        # 오프셋 기반 역순 치환으로 위치 변동 방지
        replacements = []
        for match_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                replacements.append((match.start(), match.end(), self._mask_value(match.group())))
        # 뒤에서부터 치환하여 앞쪽 오프셋 유지
        replacements.sort(key=lambda x: x[0], reverse=True)
        chars = list(text)
        for start, end, masked_val in replacements:
            chars[start:end] = list(masked_val)
        return "".join(chars)

    def _mask_value(self, value: str) -> str:
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
