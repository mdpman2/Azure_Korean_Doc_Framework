"""근거 추출 및 근거 기반 답변 생성 모듈.

검색된 문서에서 질문과 직접 관련된 근거 문장을 먼저 추출하고,
추출된 근거만을 사용하여 답변을 생성합니다.
규정형(regulatory) 및 추출형(extraction) 질문에서
정확한 수치/사실을 보존하는 것이 핵심 목적입니다.

두 가지 모드를 지원합니다:
  - extract_short_answer: 문서에서 답만 직접 추출 (단답형)
  - extract_and_answer: 근거 문장 추출 + 추론 답변 (규정형)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.schema import SearchResult
from ..core.multi_model_manager import MultiModelManager


@dataclass
class EvidenceResult:
    """근거 추출 결과.

    Attributes:
        answer: 최종 답변 텍스트.
        evidence_sentences: 문서에서 추출된 근거 문장 목록.
        sources: 근거 문장이 매칭된 출처 목록 (중복 제거).
    """
    answer: str
    evidence_sentences: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


class EvidenceExtractor:
    """근거 우선 추출 후 답변 생성기.

    검색 문서에서 근거 문장을 먼저 추출하고, 추출된 근거만으로
    답변을 생성하여 할루시네이션과 수치 왜곡을 방지합니다.
    근거 문장은 원본 문서와의 텍스트 매칭으로 출처를 추적합니다.
    """

    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager

    def extract_short_answer(
        self,
        query: str,
        documents: List[SearchResult],
        model_key: Optional[str] = None,
    ) -> Optional[EvidenceResult]:
        docs_text = self._format_documents(documents)
        prompt = (
            "다음 문서에서 질문의 답만 짧고 정확하게 추출하세요.\n"
            "규칙:\n"
            "1. 문서에 적힌 표현을 그대로 사용하세요.\n"
            "2. 설명을 붙이지 마세요.\n"
            "3. 답을 찾지 못하면 [답변] 찾을 수 없습니다 라고 답하세요.\n\n"
            f"[문서]\n{docs_text}\n\n"
            f"[질문]\n{query}\n\n"
            "아래 형식을 정확히 지키세요.\n"
            "[근거]\n문장1\n\n[답변]\n최종 답변"
        )
        return self._run(prompt, documents, model_key=model_key)

    def extract_and_answer(
        self,
        query: str,
        documents: List[SearchResult],
        model_key: Optional[str] = None,
    ) -> Optional[EvidenceResult]:
        docs_text = self._format_documents(documents)
        prompt = (
            "다음 문서를 바탕으로 질문에 답하세요.\n"
            "규칙:\n"
            "1. 먼저 질문과 직접 관련된 근거 문장 1~3개를 문서에서 그대로 추출하세요.\n"
            "2. 숫자, 금액, 기간, 횟수는 절대로 바꾸지 마세요.\n"
            "3. 답변은 추출한 근거만 사용하여 작성하세요.\n"
            "4. 답을 찾지 못하면 [답변] 찾을 수 없습니다 라고 답하세요.\n\n"
            f"[문서]\n{docs_text}\n\n"
            f"[질문]\n{query}\n\n"
            "아래 형식을 정확히 지키세요.\n"
            "[근거]\n문장1\n문장2\n\n[답변]\n최종 답변"
        )
        return self._run(prompt, documents, model_key=model_key)

    def _run(
        self,
        prompt: str,
        documents: List[SearchResult],
        model_key: Optional[str] = None,
    ) -> Optional[EvidenceResult]:
        response = self.model_manager.get_completion(
            prompt=prompt,
            model_key=model_key,
            system_message="당신은 문서 근거 추출 전문가입니다.",
            temperature=0.0,
            max_tokens=900,
        )
        if not response or response.startswith("❌"):
            return None
        return self._parse_response(response, documents)

    def _parse_response(self, response: str, documents: List[SearchResult]) -> EvidenceResult:
        evidence_match = re.search(r"\[근거\]\s*(.*?)\s*\[답변\]", response, re.DOTALL)
        answer_match = re.search(r"\[답변\]\s*(.*)$", response, re.DOTALL)

        evidence_block = evidence_match.group(1).strip() if evidence_match else ""
        answer = answer_match.group(1).strip() if answer_match else response.strip()
        evidence_sentences = [line.strip("- •\t ") for line in evidence_block.splitlines() if line.strip()]
        sources = self._match_sources(evidence_sentences, documents)

        return EvidenceResult(answer=answer, evidence_sentences=evidence_sentences, sources=sources)

    def _match_sources(self, evidence_sentences: List[str], documents: List[SearchResult]) -> List[str]:
        matched_sources = []
        normalized_documents = [
            (doc.source, self._normalize_text(doc.content))
            for doc in documents
        ]

        for sentence in evidence_sentences:
            normalized_sentence = self._normalize_text(sentence)
            if not normalized_sentence:
                continue
            for source, normalized_content in normalized_documents:
                if normalized_sentence in normalized_content:
                    matched_sources.append(source)
                    break

        if matched_sources:
            return list(dict.fromkeys(matched_sources))
        return list(dict.fromkeys(doc.source for doc in documents if doc.source))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _format_documents(self, documents: List[SearchResult]) -> str:
        formatted = []
        for index, doc in enumerate(documents, start=1):
            formatted.append(f"[{index}] 출처: {doc.source}\n{doc.content}")
        return "\n\n".join(formatted)
