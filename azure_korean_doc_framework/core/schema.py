from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Document:
    """
    LangChain의 Document 클래스를 대체하는 단순 데이터 클래스입니다.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    content: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStep:
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerArtifacts:
    answer: str
    contexts: List[str] = field(default_factory=list)
    steps: List[PipelineStep] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    gate_reason: Optional[str] = None
