from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    """
    LangChain의 Document 클래스를 대체하는 단순 데이터 클래스입니다.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
