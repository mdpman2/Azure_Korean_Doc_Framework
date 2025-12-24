import re
import tiktoken
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..core.schema import Document
from ..core.multi_model_manager import MultiModelManager

from ..config import Config


@dataclass
class ChunkingConfig:
    """청킹 설정을 관리하는 데이터 클래스"""
    min_tokens: int = 100          # 최소 토큰 수
    max_tokens: int = 500          # 최대 토큰 수
    target_tokens: int = 300       # 목표 토큰 수
    overlap_tokens: int = 50       # 오버랩 토큰 수 (약 10-15%)
    encoding_name: str = "cl100k_base"  # tiktoken 인코딩 (GPT-4, text-embedding-ada-002용)


class ChunkingStrategy(Enum):
    LEGAL = "legal"
    HIERARCHICAL = "hierarchical"
    TABULAR = "tabular"
    FALLBACK = "fallback"


class AdaptiveChunker:
    """
    문서의 특성(파일명, 내용 구조)에 따라 최적의 청킹 전략을 동적으로 선택하는 Chunker.

    개선된 기능:
    - 토큰 기반 청크 크기 제어
    - 청크 간 오버랩으로 문맥 연속성 보장
    - 한국어 문장 경계 인식
    - 강화된 메타데이터
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

        # tiktoken 인코더 초기화
        self.encoder = tiktoken.get_encoding(self.config.encoding_name)

        # MultiModelManager 초기화 (Contextual Retrieval용)
        self.model_manager = MultiModelManager()

    # ==================== 토큰 관련 유틸리티 ====================

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산합니다."""
        return len(self.encoder.encode(text))

    def _split_korean_sentences(self, text: str) -> List[str]:
        """
        한국어 텍스트를 문장 단위로 분리합니다.
        kss 라이브러리를 우선 사용하고, 실패 시 정규식 기반 분리를 사용합니다.
        """
        try:
            import kss
            sentences = kss.split_sentences(text)
            return sentences
        except Exception:
            # Fallback: 정규식 기반 한국어 문장 분리
            # 마침표, 물음표, 느낌표 뒤에 공백이나 줄바꿈이 오는 경우 분리
            pattern = r'(?<=[.!?。？！])\s+'
            sentences = re.split(pattern, text)
            return [s.strip() for s in sentences if s.strip()]

    def _merge_sentences_to_chunks(
        self,
        sentences: List[str],
        overlap_sentences: int = 1
    ) -> List[str]:
        """
        문장들을 토큰 제한에 맞게 청크로 병합합니다.
        오버랩을 적용하여 문맥 연속성을 보장합니다.
        """
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # 현재 청크에 추가 가능한지 확인
            if current_token_count + sentence_tokens <= self.config.max_tokens:
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
            else:
                # 현재 청크 저장
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))

                # 오버랩 적용: 마지막 N개 문장을 다음 청크에 포함
                overlap_start = max(0, len(current_chunk_sentences) - overlap_sentences)
                overlap_sents = current_chunk_sentences[overlap_start:]

                # 새 청크 시작
                current_chunk_sentences = overlap_sents + [sentence]
                current_token_count = sum(self._count_tokens(s) for s in current_chunk_sentences)

        # 마지막 청크 저장
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _split_with_overlap(self, text: str) -> List[str]:
        """
        텍스트를 한국어 문장 단위로 분리 후 오버랩을 적용하여 청킹합니다.
        """
        if not text or not text.strip():
            return []

        # 1. 한국어 문장 분리
        sentences = self._split_korean_sentences(text)

        if not sentences:
            return [text] if self._count_tokens(text) <= self.config.max_tokens else []

        # 2. 오버랩 적용하여 청크 생성
        return self._merge_sentences_to_chunks(sentences, overlap_sentences=2)

    def _enrich_metadata(
        self,
        base_metadata: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        chunk_text: str,
        section_title: str = ""
    ) -> Dict[str, Any]:
        """청크에 강화된 메타데이터를 추가합니다."""
        enriched = base_metadata.copy()
        enriched.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "token_count": self._count_tokens(chunk_text),
            "char_count": len(chunk_text),
            "section_title": section_title,
        })
        return enriched

    # ==================== 메인 청킹 로직 ====================

    def chunk(self, segments: List[Dict[str, Any]], filename: str = "", extra_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Main Entrypoint: 문서 세그먼트를 입력받아 적절한 전략으로 청킹을 수행합니다.
        """
        if extra_metadata is None: extra_metadata = {}

        # 1. 문서 분류
        strategy = self._classify_document(filename, segments)
        print(f"🔍 Document Classification: '{filename}' -> {strategy.name}")
        print(f"   ⚙️ Config: min={self.config.min_tokens}, max={self.config.max_tokens}, overlap={self.config.overlap_tokens} tokens")

        # 2. 전략 실행 (Dispatcher)
        if strategy == ChunkingStrategy.LEGAL:
            chunks = self._chunk_legal(segments, extra_metadata, filename=filename)
        elif strategy == ChunkingStrategy.TABULAR:
            chunks = self._chunk_tabular(segments, extra_metadata)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            chunks = self._chunk_hierarchical(segments, extra_metadata)
        else:
            chunks = self._chunk_fallback(segments, extra_metadata)

        # 3. 최종 메타데이터 강화
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata = self._enrich_metadata(
                chunk.metadata,
                chunk_index=i,
                total_chunks=total,
                chunk_text=chunk.page_content,
                section_title=chunk.metadata.get("breadcrumb", "")
            )

        print(f"   ✅ Generated {total} chunks")
        return chunks

    def _classify_document(self, filename: str, segments: List[Dict[str, Any]]) -> ChunkingStrategy:
        """파일명과 콘텐츠 비율을 기반으로 청킹 전략을 결정합니다."""
        name = filename.lower()

        # 1. Legal Strategy (파일명 or 특정 키워드)
        if any(k in name for k in ["[민사]", "[형사]", "[행정]", "[특허]", "판례"]):
            return ChunkingStrategy.LEGAL

        # 2. Tabular Strategy (파일명 or 표 비중)
        if any(k in name for k in ["재정동향", "통화신용정책", "현황"]):
            return ChunkingStrategy.TABULAR

        # 표가 전체 세그먼트의 50% 이상이면 Tabular로 간주
        table_count = sum(1 for s in segments if s['type'] == 'table')
        if len(segments) > 0 and (table_count / len(segments)) > 0.5:
            return ChunkingStrategy.TABULAR

        # 3. Hierarchical Strategy (기본 보고서)
        header_count = sum(1 for s in segments if s['type'] == 'header')
        if header_count > 0:
            return ChunkingStrategy.HIERARCHICAL

        # 4. Fallback
        return ChunkingStrategy.FALLBACK

    def _generate_context(self, chunk_text: str, filename: str) -> str:
        """Contextual Retrieval: 청크의 문맥을 LLM을 통해 생성합니다."""
        try:
            system_prompt = (
                "You are a legal expert assistant.\n"
                f"The following is a section from a document named '{filename}'.\n"
                "Please read the section and explain its specific context in 2-3 sentences.\n"
                "Focus on identifying the legal principle, the court's reasoning, or the specific subject matter being discussed.\n"
                "Do not just repeat the text, but contextualize it so it can be understood in isolation."
            )

            user_prompt = f"Section Content:\n{chunk_text}\n\nContext Explanation:"

            context = self.model_manager.get_completion(
                prompt=user_prompt,
                system_message=system_prompt,
                model_key="gpt-4o", # 기본적으로 고성능 모델 사용
                temperature=0
            )
            return context.strip()
        except Exception as e:
            print(f"⚠️ Context Generation Failed: {e}")
            return ""

    def _chunk_legal(self, segments: List[Dict[str, Any]], extra_metadata: Dict[str, Any], filename: str) -> List[Document]:
        """Strategy A: Regex-based split for Legal documents + Contextual Retrieval"""
        print("   ⚖️ Strategy: LEGAL (Regex Split + Contextual Retrieval + Overlap)")

        # 1. 1차 통합
        full_text = "\n\n".join([s['content'] for s in segments])

        # 2. Regex Split (판례 구조 기반 - 【주문】, 【이유】 등)
        split_pattern = r"(?=【.*?】)"
        raw_chunks = re.split(split_pattern, full_text)

        final_chunks = []
        print(f"      👉 Splitting into {len(raw_chunks)} raw blocks...")

        for i, raw_text in enumerate(raw_chunks):
            if not raw_text.strip(): continue

            # 토큰 수 체크 - 너무 크면 추가 분할
            if self._count_tokens(raw_text) > self.config.max_tokens:
                sub_chunks = self._split_with_overlap(raw_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    meta = extra_metadata.copy()
                    meta['strategy'] = 'legal_contextual'
                    meta['sub_chunk'] = f"{i+1}.{j+1}"
                    final_chunks.append(Document(page_content=sub_chunk, metadata=meta))
            else:
                # 3. Contextual Retrieval (LLM 호출) - 비용 절감을 위해 첫 5개만
                context = ""
                if i < 5:
                    print(f"      🤖 Generating context for chunk {i+1}...")
                    context = self._generate_context(raw_text[:1000], filename)

                content_with_context = raw_text
                if context:
                    content_with_context = f"[Context: {context}]\n\n{raw_text}"

                meta = extra_metadata.copy()
                meta['strategy'] = 'legal_contextual'
                final_chunks.append(Document(page_content=content_with_context, metadata=meta))

        return final_chunks

    def _chunk_tabular(self, segments: List[Dict[str, Any]], extra_metadata: Dict[str, Any]) -> List[Document]:
        """Strategy C: Row-wise serialization for Data/Table heavy documents"""
        print("   📊 Strategy: TABULAR (Row-wise Serialization + Token Control)")
        final_chunks = []

        for seg in segments:
            if seg['type'] == 'table':
                # 마크다운 표 -> 자연어 문장 변환
                sentences = self._markdown_table_to_sentences(seg['content'])

                serialized_text = "\n".join(sentences)
                if not serialized_text:
                    serialized_text = seg['content']  # 실패 시 원문

                # 토큰 수 체크 - 너무 크면 분할
                if self._count_tokens(serialized_text) > self.config.max_tokens:
                    sub_chunks = self._split_with_overlap(serialized_text)
                    for sub_chunk in sub_chunks:
                        meta = extra_metadata.copy()
                        meta['is_table_data'] = True
                        meta["page"] = seg.get("page", 1)
                        final_chunks.append(Document(page_content=sub_chunk, metadata=meta))
                else:
                    meta = extra_metadata.copy()
                    meta['is_table_data'] = True
                    meta["page"] = seg.get("page", 1)
                    final_chunks.append(Document(page_content=serialized_text, metadata=meta))

            else:
                # 일반 텍스트는 오버랩 청킹
                text_content = seg['content'].strip()
                if text_content and len(text_content) > 10:
                    sub_chunks = self._split_with_overlap(text_content)
                    for sub_chunk in sub_chunks:
                        if self._count_tokens(sub_chunk) >= self.config.min_tokens:
                            final_chunks.append(Document(page_content=sub_chunk, metadata=extra_metadata.copy()))

        return final_chunks

    def _markdown_table_to_sentences(self, markdown_table: str) -> List[str]:
        """Markdown 테이블을 '헤더는 값이다' 형태의 문장으로 변환합니다."""
        lines = markdown_table.strip().split('\n')
        if len(lines) < 3: return []

        header_line = lines[0]
        data_lines = lines[2:]  # 구분선 건너김

        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        sentences = []

        for row in data_lines:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if not cells: continue

            row_parts = []
            for h, c in zip(headers, cells):
                if h and c:
                    row_parts.append(f"{h}은(는) {c}")

            if row_parts:
                sentences.append(", ".join(row_parts) + ".")

        return sentences

    def _chunk_hierarchical(self, segments: List[Dict[str, Any]], extra_metadata: Dict[str, Any]) -> List[Document]:
        """Strategy B: Context-Rich Rolling Window with Overlap"""
        print("   🌲 Strategy: HIERARCHICAL (Context-Rich + Overlap)")
        final_chunks = []
        header_stack = []  # [(level, text), ...]
        text_buffer = []

        def get_breadcrumb():
            return " > ".join([h[1] for h in header_stack])

        def flush_text_buffer():
            if not text_buffer: return
            combined_text = "\n\n".join(text_buffer)
            text_buffer.clear()

            current_breadcrumb = get_breadcrumb()

            if not combined_text or len(combined_text) < 10:
                return

            # 오버랩 적용 청킹
            sub_chunks = self._split_with_overlap(combined_text)

            for sub_chunk in sub_chunks:
                if self._count_tokens(sub_chunk) < self.config.min_tokens:
                    continue

                base_meta = extra_metadata.copy()
                base_meta["breadcrumb"] = current_breadcrumb
                base_meta["type"] = "text"

                content = f"[{current_breadcrumb}]\n{sub_chunk}" if current_breadcrumb else sub_chunk
                final_chunks.append(Document(page_content=content, metadata=base_meta))

        for seg in segments:
            seg_type = seg["type"]
            content = seg["content"]

            if seg_type == "header":
                flush_text_buffer()
                level = 0
                clean_header = content.strip()
                if clean_header.startswith("#"):
                    level = len(clean_header.split()[0])
                    clean_header = clean_header.lstrip("#").strip()
                else:
                    level = 1

                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, clean_header))

            elif seg_type == "table":
                flush_text_buffer()
                current_breadcrumb = get_breadcrumb()

                # 표 직렬화
                sentences = self._markdown_table_to_sentences(content)
                serialized = "\n".join(sentences) if sentences else content

                full_content = f"[{current_breadcrumb}]\n{serialized}" if current_breadcrumb else serialized

                meta = extra_metadata.copy()
                meta["breadcrumb"] = current_breadcrumb
                meta["type"] = "table"
                meta["page"] = seg.get("page", 1)
                final_chunks.append(Document(page_content=full_content, metadata=meta))

            elif seg_type in ["text", "image"]:
                if content.strip():
                    text_buffer.append(content)

        flush_text_buffer()
        return final_chunks

    def _chunk_fallback(self, segments: List[Dict[str, Any]], extra_metadata: Dict[str, Any]) -> List[Document]:
        """Strategy D: Simple Fallback with Overlap"""
        print("   🍂 Strategy: FALLBACK (Overlap Chunking)")
        all_text = "\n\n".join([s['content'] for s in segments if s.get('content', '').strip()])

        if not all_text or len(all_text.strip()) < 10:
            print("   ⚠️ No content to chunk")
            return []

        # 오버랩 적용 청킹
        sub_chunks = self._split_with_overlap(all_text)

        final_chunks = []
        for sub_chunk in sub_chunks:
            if self._count_tokens(sub_chunk) >= self.config.min_tokens:
                final_chunks.append(Document(page_content=sub_chunk, metadata=extra_metadata.copy()))

        return final_chunks


# Backward Compatibility
KoreanSemanticChunker = AdaptiveChunker
