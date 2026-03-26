"""
Context-Rich Rolling Window 청킹 모듈 + Contextual Retrieval

문서의 구조(Hierarchy)와 문맥(Context)을 보존하는 적응형 청킹 시스템.
kss 한국어 문장 분리 + tiktoken 토큰 기반 분할.

[2026-02 v4.1 업데이트 - Contextual Retrieval (Anthropic 방식)]
- _apply_contextual_retrieval(): 모든 청크에 LLM 기반 문서 맥락 자동 추가
- _generate_context(): 전체 문서 참조하여 청크별 맥락 생성 (Anthropic 프롬프트)
- Contextual BM25: 맥락 포함 텍스트로 BM25 키워드 검색 정확도 향상
- Contextual Embeddings: 맥락 포함 텍스트로 벡터 임베딩 정확도 향상
- 검색 실패율 49% 감소 (BM25 + Vector + Semantic 결합 시)

[2026-02 v4.0]
- 엔티티 인식 메타데이터 (hangul_ratio, graph_rag_eligible)
- 한글 비율 계산 (사전 컴파일된 _HANGUL_SYLLABLE_RE)
- _enrich_metadata(): Graph RAG 적격 여부 태깅
"""

import re
import tiktoken
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..core.schema import Document
from ..core.multi_model_manager import MultiModelManager

from ..config import Config

# kss 모듈 사전 임포트 (매번 try/except 방지)
try:
    import kss as _kss_module
    _HAS_KSS = True
except ImportError:
    _kss_module = None
    _HAS_KSS = False

# 사전 컴파일된 한글 패턴 (성능 최적화)
_HANGUL_SYLLABLE_RE = re.compile(r'[\uAC00-\uD7AF]')


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

    [v4.1 업데이트 - Contextual Retrieval]
    - Contextual Retrieval (Anthropic 방식): 모든 청크에 LLM 맥락 자동 추가
    - 전체 문서 참조하여 청크별 간결한 맥락 생성 (50-150 토큰)
    - 맥락 포함 텍스트로 BM25 + 벡터 검색 동시 개선
    - 원본/맥락 분리 저장: 검색은 맥락 포함, 답변은 원본 사용

    [v4.0 업데이트]
    - 엔티티 인식 청킹 (LangExtract 기반 엔티티 경계 보존)
    - 한국어 Unicode 토크나이저 연동
    - Graph RAG용 엔티티 메타데이터 태깅
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

        # tiktoken 인코더 초기화
        self.encoder = tiktoken.get_encoding(self.config.encoding_name)

        # 반복 토큰 계산 캐시
        self._token_count_cache: Dict[str, int] = {}

        # Contextual Retrieval을 사용할 때만 LLM 매니저 초기화
        self.model_manager = MultiModelManager() if Config.CONTEXTUAL_RETRIEVAL_ENABLED else None

    # ==================== 토큰 관련 유틸리티 ====================

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산합니다."""
        cached = self._token_count_cache.get(text)
        if cached is not None:
            return cached

        token_count = len(self.encoder.encode(text))
        self._token_count_cache[text] = token_count
        return token_count

    # 정규식 기반 문장 분리 패턴 (사전 컴파일)
    _SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?。？！])\s+')

    def _split_korean_sentences(self, text: str) -> List[str]:
        """
        한국어 텍스트를 문장 단위로 분리합니다.
        kss 라이브러리를 우선 사용하고, 실패 시 정규식 기반 분리를 사용합니다.
        """
        if _HAS_KSS:
            try:
                return _kss_module.split_sentences(text)
            except Exception:
                pass
        # Fallback: 정규식 기반 한국어 문장 분리
        sentences = self._SENTENCE_SPLIT_RE.split(text)
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
        current_chunk_token_counts = []
        current_token_count = 0
        sentence_token_counts = [self._count_tokens(sentence) for sentence in sentences]

        for sentence, sentence_tokens in zip(sentences, sentence_token_counts):

            # 현재 청크에 추가 가능한지 확인
            if current_token_count + sentence_tokens <= self.config.max_tokens:
                current_chunk_sentences.append(sentence)
                current_chunk_token_counts.append(sentence_tokens)
                current_token_count += sentence_tokens
            else:
                # 현재 청크 저장
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))

                # 오버랩 적용: 마지막 N개 문장을 다음 청크에 포함
                overlap_start = max(0, len(current_chunk_sentences) - overlap_sentences)
                overlap_sents = current_chunk_sentences[overlap_start:]
                overlap_token_counts = current_chunk_token_counts[overlap_start:]
                overlap_token_count = sum(overlap_token_counts)

                # 새 청크 시작
                current_chunk_sentences = overlap_sents + [sentence]
                current_chunk_token_counts = overlap_token_counts + [sentence_tokens]
                current_token_count = overlap_token_count + sentence_tokens

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
        """청크에 강화된 메타데이터를 추가합니다. (v4.0: 엔티티 태깅 포함)"""
        enriched = base_metadata.copy()
        enriched.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "token_count": self._count_tokens(chunk_text),
            "char_count": len(chunk_text),
            "section_title": section_title,
            # v4.0: 한국어 텍스트 비율 메타데이터
            "hangul_ratio": self._calculate_hangul_ratio(chunk_text),
        })
        if "type" in enriched and "chunk_type" not in enriched:
            enriched["chunk_type"] = enriched["type"]
        if "page" in enriched and "page_number" not in enriched:
            enriched["page_number"] = enriched["page"]
        if "source" in enriched and "source_file" not in enriched:
            enriched["source_file"] = enriched["source"]
        return enriched

    def _calculate_hangul_ratio(self, text: str) -> float:
        """텍스트의 한글 비율을 계산합니다. (v4.0 신규)"""
        if not text:
            return 0.0
        hangul_count = len(_HANGUL_SYLLABLE_RE.findall(text))
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        return round(hangul_count / max(total_chars, 1), 3)

    def _collect_source_regions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """세그먼트에 포함된 source_regions를 병합하고 중복 제거합니다."""
        merged_regions = []
        seen = set()

        for segment in segments:
            regions = segment.get("source_regions") or []
            if not regions and (segment.get("bounding_box") or segment.get("polygon")):
                regions = [{
                    "page_number": segment.get("page"),
                    "bounding_box": segment.get("bounding_box"),
                    "polygon": segment.get("polygon"),
                    "unit": segment.get("page_unit"),
                }]

            for region in regions:
                polygon = region.get("polygon") or []
                polygon_key = tuple((point.get("x"), point.get("y")) for point in polygon)
                key = (region.get("page_number"), polygon_key)
                if key in seen:
                    continue
                seen.add(key)
                merged_regions.append(region)

        merged_regions.sort(key=lambda item: (item.get("page_number") or 0, (item.get("bounding_box") or {}).get("top", 0)))
        return merged_regions

    def _apply_layout_metadata(
        self,
        metadata: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """세그먼트의 레이아웃 메타데이터를 청크 메타데이터로 전파합니다."""
        enriched = metadata.copy()
        source_regions = self._collect_source_regions(segments)
        if not source_regions:
            return enriched

        enriched["source_regions"] = source_regions

        page_numbers = sorted({region.get("page_number") for region in source_regions if region.get("page_number") is not None})
        if page_numbers:
            enriched.setdefault("page", page_numbers[0])
            enriched["page_numbers"] = page_numbers

        primary_region = source_regions[0]
        if primary_region.get("bounding_box") is not None:
            enriched["bounding_box"] = primary_region["bounding_box"]
        if primary_region.get("polygon"):
            enriched["polygon"] = primary_region["polygon"]
        if primary_region.get("unit") is not None:
            enriched["page_unit"] = primary_region["unit"]

        return enriched

    # ==================== 표/이미지 청크 분리 ====================

    def _extract_special_chunks(
        self,
        segments: List[Dict[str, Any]],
        extra_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        표(table)와 이미지(image) 세그먼트를 별도 청크로 분리합니다.
        검색 최적화를 위해 독립적인 청크로 생성합니다.
        """
        special_chunks = []

        for seg in segments:
            seg_type = seg.get("type", "")
            content = seg.get("content", "").strip()

            if not content:
                continue

            if seg_type == "table":
                # 표를 별도 청크로 생성
                meta = extra_metadata.copy()
                meta.update({
                    "type": "table",
                    "is_table_data": True,
                    "page": seg.get("page", 1),
                    "searchable": True  # 검색 최적화 플래그
                })
                meta = self._apply_layout_metadata(meta, [seg])
                special_chunks.append(Document(page_content=content, metadata=meta))

            elif seg_type == "image":
                # 이미지 설명을 별도 청크로 생성
                meta = extra_metadata.copy()
                meta.update({
                    "type": "image_description",
                    "is_image_data": True,
                    "page": seg.get("page", 1),
                    "searchable": True
                })
                meta = self._apply_layout_metadata(meta, [seg])
                special_chunks.append(Document(page_content=content, metadata=meta))

        return special_chunks

    # ==================== 메인 청킹 로직 ====================

    def chunk(self, segments: List[Dict[str, Any]], filename: str = "", extra_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Main Entrypoint: 문서 세그먼트를 입력받아 적절한 전략으로 청킹을 수행합니다.

        개선 사항:
        - 표/이미지/번호목록 구조 보존
        - 유형별 청크 분리 생성
        - 검색 최적화를 위한 메타데이터 강화
        """
        if extra_metadata is None: extra_metadata = {}

        # 0. 표/이미지 세그먼트 별도 청크로 분리 (검색 최적화)
        table_image_chunks = self._extract_special_chunks(segments, extra_metadata)
        print(f"   📊 Extracted {len(table_image_chunks)} table/image chunks for search optimization")

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

        # 2.5. 표/이미지 별도 청크 병합
        chunks.extend(table_image_chunks)

        # 2.6. Contextual Retrieval: 모든 청크에 문서 맥락 추가 (Anthropic 방식)
        if Config.CONTEXTUAL_RETRIEVAL_ENABLED:
            full_document_text = "\n\n".join([s.get('content', '') for s in segments])
            chunks = self._apply_contextual_retrieval(chunks, filename, full_document_text)

        # 3. 최종 메타데이터 강화 + Graph RAG 플래그 (단일 루프로 통합)
        _numbered_section_re = re.compile(r'###?\s*\d{2}\.')
        total = len(chunks)
        table_count = 0
        image_count = 0

        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            is_table = chunk.metadata.get('is_table_data', False)
            is_image = chunk.metadata.get('is_image_data', False)

            # 메타데이터 강화
            chunk.metadata = self._enrich_metadata(
                chunk.metadata,
                chunk_index=i,
                total_chunks=total,
                chunk_text=content,
                section_title=chunk.metadata.get("breadcrumb", "")
            )

            # 청크 유형별 추가 정보 (조건 통합)
            if is_table:
                table_count += 1
            elif is_image:
                image_count += 1
            else:
                # v4.0: Graph RAG 대상 플래그 (텍스트 청크만)
                chunk.metadata['graph_rag_eligible'] = True

            if '> **[이미지/차트 설명' in content:
                chunk.metadata['contains_image_desc'] = True
            if _numbered_section_re.search(content):
                chunk.metadata['contains_numbered_section'] = True
            if '|' in content and '---' in content:
                chunk.metadata['contains_table'] = True

        text_count = total - table_count - image_count
        print(f"   ✅ Generated {total} chunks")
        print(f"      - Text chunks: {text_count}")
        print(f"      - Table chunks: {table_count}")
        print(f"      - Image chunks: {image_count}")

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

    def _generate_context(self, chunk_text: str, filename: str, full_document_text: str = "", _doc_preview: str = "") -> str:
        """
        Contextual Retrieval (Anthropic 방식): 청크의 문맥을 LLM을 통해 생성합니다.

        전체 문서 맥락을 참조하여 각 청크가 독립적으로도 이해될 수 있도록
        간결한 맥락 설명을 생성합니다.

        Args:
            chunk_text: 청크 텍스트
            filename: 문서 파일명
            full_document_text: 전체 문서 텍스트 (맥락 참조용)
            _doc_preview: 사전 생성된 문서 프리뷰 (호출자가 전달 시 재계산 방지)

        Returns:
            청크의 맥락 설명 텍스트
        """
        if not self.model_manager:
            return ""

        try:
            # 전체 문서가 너무 길면 앞부분만 사용 (토큰 절약)
            doc_preview = _doc_preview or (full_document_text[:8000] if full_document_text else "")

            system_prompt = (
                "당신은 문서 분석 전문가입니다. "
                "청크의 검색 검색을 개선하기 위해 전체 문서 내에서 이 청크를 위치시키는 "
                "짧고 간결한 맥락을 한국어로 제공해 주세요.\n"
                "규칙:\n"
                "1. 간결한 맥락만 답하고 다른 것은 없습니다.\n"
                "2. 2-3문장으로 청크가 어떤 문서의 어떤 부분인지, 핵심 주제가 무엇인지 설명하세요.\n"
                "3. 문서명, 관련 엔티티(회사명, 인명, 제품명 등), 기간 정보를 반드시 포함하세요.\n"
                "4. 청크 텍스트를 반복하지 말고, 맥락적 정보만 제공하세요."
            )

            user_prompt = (
                f"<document>\n{doc_preview}\n</document>\n\n"
                f"여기에 전체 문서 내에서 위치시키고자 하는 청크가 있습니다:\n"
                f"<chunk>\n{chunk_text[:2000]}\n</chunk>\n\n"
                f"문서명: {filename}\n"
                f"청크의 검색 검색을 개선하기 위해 전체 문서 내에서 이 청크를 위치시키는 "
                f"짧고 간결한 맥락을 한국어로 제공해 주세요. 간결한 맥락만 답하고 다른 것은 없습니다."
            )

            context = self.model_manager.get_completion(
                prompt=user_prompt,
                system_message=system_prompt,
                model_key=Config.CONTEXTUAL_RETRIEVAL_MODEL,
                temperature=0,
                max_tokens=Config.CONTEXTUAL_RETRIEVAL_MAX_TOKENS
            )
            return context.strip()
        except Exception as e:
            print(f"⚠️ Context Generation Failed: {e}")
            return ""

    def _apply_contextual_retrieval(
        self,
        chunks: List[Document],
        filename: str,
        full_document_text: str
    ) -> List[Document]:
        """
        Contextual Retrieval (Anthropic 방식): 모든 청크에 문서 맥락을 앞에 추가합니다.

        맥락이 추가된 청크는:
        - BM25 키워드 검색 시 맥락 정보로 정확도 향상 (Contextual BM25)
        - 벡터 임베딩 시 맥락 정보 포함으로 유사성 검색 향상 (Contextual Embeddings)

        Args:
            chunks: 청크 리스트
            filename: 문서 파일명
            full_document_text: 전체 문서 텍스트

        Returns:
            맥락이 추가된 청크 리스트
        """
        if not self.model_manager:
            return chunks

        indexed_text_chunks = [
            (index, chunk)
            for index, chunk in enumerate(chunks)
            if not chunk.metadata.get('is_table_data') and not chunk.metadata.get('is_image_data')
        ]

        if not indexed_text_chunks:
            return chunks

        batch_size = Config.CONTEXTUAL_RETRIEVAL_BATCH_SIZE
        total = len(indexed_text_chunks)
        print(f"   🧠 Contextual Retrieval: {total}개 텍스트 청크에 맥락 추가 중... (batch_size={batch_size})")

        doc_preview = full_document_text[:8000] if full_document_text else ""

        def _generate_and_apply(indexed_chunk):
            """단일 청크에 맥락을 생성하고 적용하는 워커 함수"""
            idx, chunk = indexed_chunk
            context = self._generate_context(
                chunk_text=chunk.page_content,
                filename=filename,
                full_document_text=full_document_text,
                _doc_preview=doc_preview
            )
            return idx, context

        contexts_by_index: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(_generate_and_apply, indexed_chunk): indexed_chunk[0]
                for indexed_chunk in indexed_text_chunks
            }
            completed = 0
            for future in as_completed(futures):
                idx, context = future.result()
                contexts_by_index[idx] = context
                completed += 1
                if completed % batch_size == 0 or completed == total:
                    print(f"      📝 맥락 생성 완료: {completed}/{total}")

        success_count = 0
        for index, chunk in indexed_text_chunks:
            context = contexts_by_index.get(index)
            if context:
                chunk.metadata['original_chunk'] = chunk.page_content
                chunk.metadata['context'] = context
                chunk.metadata['has_contextual_retrieval'] = True
                chunk.page_content = f"[맥락: {context}]\n\n{chunk.page_content}"
                success_count += 1
            else:
                chunk.metadata['has_contextual_retrieval'] = False

        print(f"   ✅ Contextual Retrieval 완료: {success_count}/{total}개 청크에 맥락 추가")
        return chunks

    def _chunk_legal(self, segments: List[Dict[str, Any]], extra_metadata: Dict[str, Any], filename: str) -> List[Document]:
        """Strategy A: Regex-based split for Legal documents"""
        print("   \u2696\ufe0f Strategy: LEGAL (Regex Split + Overlap)")

        # 1. 1\ucc28 \ud1b5\ud569
        full_text = "\n\n".join([s['content'] for s in segments])

        # 2. Regex Split (\ud310\ub840 \uad6c\uc870 \uae30\ubc18 - \u3010\uc8fc\ubb38\u3011, \u3010\uc774\uc720\u3011 \ub4f1)
        split_pattern = r"(?=\u3010.*?\u3011)"
        raw_chunks = re.split(split_pattern, full_text)

        final_chunks = []
        print(f"      \ud83d\udc49 Splitting into {len(raw_chunks)} raw blocks...")

        for i, raw_text in enumerate(raw_chunks):
            if not raw_text.strip(): continue

            # \ud1a0\ud070 \uc218 \uccb4\ud06c - \ub108\ubb34 \ud06c\uba74 \ucd94\uac00 \ubd84\ud560
            if self._count_tokens(raw_text) > self.config.max_tokens:
                sub_chunks = self._split_with_overlap(raw_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    meta = extra_metadata.copy()
                    meta['strategy'] = 'legal'
                    meta['sub_chunk'] = f"{i+1}.{j+1}"
                    final_chunks.append(Document(page_content=sub_chunk, metadata=meta))
            else:
                meta = extra_metadata.copy()
                meta['strategy'] = 'legal'
                final_chunks.append(Document(page_content=raw_text, metadata=meta))

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
                        meta["type"] = "table"
                        meta = self._apply_layout_metadata(meta, [seg])
                        final_chunks.append(Document(page_content=sub_chunk, metadata=meta))
                else:
                    meta = extra_metadata.copy()
                    meta['is_table_data'] = True
                    meta["page"] = seg.get("page", 1)
                    meta["type"] = "table"
                    meta = self._apply_layout_metadata(meta, [seg])
                    final_chunks.append(Document(page_content=serialized_text, metadata=meta))

            else:
                # 일반 텍스트는 오버랩 청킹
                text_content = seg['content'].strip()
                if text_content and len(text_content) > 10:
                    sub_chunks = self._split_with_overlap(text_content)
                    for sub_chunk in sub_chunks:
                        if self._count_tokens(sub_chunk) >= self.config.min_tokens:
                            meta = extra_metadata.copy()
                            meta["type"] = seg.get("type", "text")
                            meta = self._apply_layout_metadata(meta, [seg])
                            final_chunks.append(Document(page_content=sub_chunk, metadata=meta))

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
        text_buffer: List[Dict[str, Any]] = []

        def get_breadcrumb():
            return " > ".join([h[1] for h in header_stack])

        def flush_text_buffer():
            if not text_buffer: return
            buffered_segments = text_buffer.copy()
            combined_text = "\n\n".join(seg["content"] for seg in buffered_segments)
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
                base_meta = self._apply_layout_metadata(base_meta, buffered_segments)

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
                meta = self._apply_layout_metadata(meta, [seg])
                final_chunks.append(Document(page_content=full_content, metadata=meta))

            elif seg_type in ["text", "image"]:
                if content.strip():
                    text_buffer.append(seg)

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
                meta = extra_metadata.copy()
                meta["type"] = "text"
                meta = self._apply_layout_metadata(meta, segments)
                final_chunks.append(Document(page_content=sub_chunk, metadata=meta))

        return final_chunks


# Backward Compatibility
KoreanSemanticChunker = AdaptiveChunker
