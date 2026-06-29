"""RAG 파이프라인 전체에서 공유되는 핵심 데이터 모델.

문서, 검색 결과, 파이프라인 단계, 최종 답변 산출물 등
프레임워크 전역에서 사용되는 데이터 클래스를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """
    LangChain의 Document 클래스를 대체하는 단순 데이터 클래스입니다.

    Attributes:
        page_content: 문서 텍스트 내용.
        metadata: 파일명, 페이지 번호, 청크 유형 등 부가 정보.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Azure AI Search 또는 Graph RAG에서 반환된 개별 검색 결과.

    Attributes:
        content: 검색된 문서 텍스트.
        source: 출처 식별자 (파일명 또는 parent_id).
        score: 검색 유사도/관련성 점수 (0.0~1.0).
        metadata: 추가 메타데이터 (페이지, bbox 등).
    """
    content: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStep:
    """Guardrail 파이프라인의 개별 검증 단계 결과.

    Attributes:
        name: 단계 이름 (예: 'retrieval_gate', 'faithfulness').
        passed: 해당 단계의 통과 여부.
        detail: 세부 결과 정보 (점수, 사유 등).
    """
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerArtifacts:
    """RAG 파이프라인의 최종 답변 및 진단 산출물.

    ``agent.answer_question(return_artifacts=True)`` 호출 시 반환되며,
    답변 텍스트와 함께 검색 결과, 파이프라인 단계, 진단 정보를 포함합니다.

    Attributes:
        answer: 최종 생성 답변 텍스트.
        contexts: 답변 생성에 사용된 컨텍스트 목록.
        steps: Guardrail 파이프라인 각 단계의 통과/실패 기록.
        search_results: 검색된 원본 문서 목록.
        gate_reason: Retrieval Gate 실패 시 차단 사유.
        diagnostics: 쿼리 변형 수, 상위 점수, 모델 등 운영 진단 정보.
    """
    answer: str
    contexts: List[str] = field(default_factory=list)
    steps: List[PipelineStep] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    gate_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
