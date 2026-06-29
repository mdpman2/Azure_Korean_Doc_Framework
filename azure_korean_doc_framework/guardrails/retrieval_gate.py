"""검색 품질 게이트 모듈.

검색 결과의 품질(최고 점수, 유효 문서 수)이 임계값을 충족하는지 검증합니다.
품질 미달 시 답변 생성을 차단하거나 soft-fail(경고) 처리하여
저품질 검색 결과에 기반한 부정확한 답변 생성을 방지합니다.

Usage:
    gate = RetrievalQualityGate(min_top_score=0.15, soft_mode=True)
    result = gate.evaluate(search_results)
    if not result.passed:
        print(f"검색 품질 미달: {result.reason}")
"""

from dataclasses import dataclass
from typing import List

from ..core.schema import SearchResult


@dataclass
class RetrievalGateResult:
    """검색 품질 게이트 평가 결과.

    Attributes:
        passed: 품질 기준 충족 여부.
        soft_fail: soft 모드에서의 경고 여부 (차단 대신 경고만).
        top_score: 최상위 문서의 검색 점수.
        qualifying_count: 최소 점수를 충족하는 문서 수.
        reason: 실패 시 상세 사유.
    """
    passed: bool
    soft_fail: bool = False
    top_score: float = 0.0
    qualifying_count: int = 0
    reason: str = ""


class RetrievalQualityGate:
    """검색 결과 품질 게이트.

    min_top_score, min_doc_count 등의 임계값으로 검색 품질을 판단합니다.
    soft_mode=True이면 차단 대신 경고(soft_fail)만 합니다.
    """
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
