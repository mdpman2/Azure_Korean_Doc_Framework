from dataclasses import dataclass
from typing import List

from ..core.schema import SearchResult


@dataclass
class RetrievalGateResult:
    passed: bool
    soft_fail: bool = False
    top_score: float = 0.0
    qualifying_count: int = 0
    reason: str = ""


class RetrievalQualityGate:
    def __init__(
        self,
        min_top_score: float = 0.15,
        min_doc_count: int = 1,
        min_doc_score: float = 0.05,
        soft_mode: bool = True,
    ):
        self.min_top_score = min_top_score
        self.min_doc_count = min_doc_count
        self.min_doc_score = min_doc_score
        self.soft_mode = soft_mode

    def evaluate(self, documents: List[SearchResult]) -> RetrievalGateResult:
        if not documents:
            return RetrievalGateResult(
                passed=False,
                soft_fail=self.soft_mode,
                reason="no_documents",
            )

        top_score = documents[0].score
        qualifying = [doc for doc in documents if doc.score >= self.min_doc_score]

        if top_score < self.min_top_score:
            return RetrievalGateResult(
                passed=False,
                soft_fail=self.soft_mode,
                top_score=top_score,
                qualifying_count=len(qualifying),
                reason=f"top_score({top_score:.3f}) < min({self.min_top_score:.3f})",
            )

        if len(qualifying) < self.min_doc_count:
            return RetrievalGateResult(
                passed=False,
                soft_fail=self.soft_mode,
                top_score=top_score,
                qualifying_count=len(qualifying),
                reason=f"qualifying_docs({len(qualifying)}) < min({self.min_doc_count})",
            )

        return RetrievalGateResult(
            passed=True,
            top_score=top_score,
            qualifying_count=len(qualifying),
        )
