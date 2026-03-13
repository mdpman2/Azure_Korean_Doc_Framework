"""
LangExtract-inspired 구조화된 엔티티 추출 모듈 (v4.0)

Google LangExtract(https://github.com/google/langextract)의 핵심 개념을 참조하여 구현한
한국어 문서 구조화 추출 시스템입니다.

핵심 기능:
- GPT-5.4를 활용한 Few-Shot 기반 구조화 추출
- 한국어/CJK Unicode 토크나이저 (정확한 위치 매핑)
- Multi-Pass 추출 (향상된 Recall)
- Source Grounding (원문 위치 추적)
- 병렬 처리 지원

[2026-02 v4.0 신규]
- LangExtract 기반 구조화 추출 아키텍처
- 한국어 Unicode 토크나이저
- 원문 위치 추적 (char_interval)
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import Config
from ..utils.azure_clients import AzureClientFactory


# ==================== 데이터 모델 ====================

@dataclass
class CharInterval:
    """원문에서의 문자 위치 구간"""
    start_pos: int
    end_pos: int

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos


@dataclass
class Extraction:
    """
    추출된 엔티티 (LangExtract의 Extraction 클래스 참조)

    Attributes:
        extraction_class: 엔티티 클래스 (예: "인물", "조직")
        extraction_text: 추출된 원문 텍스트
        char_interval: 원문에서의 위치
        attributes: 추가 속성 (예: {"역할": "대표이사"})
        description: 엔티티 설명
    """
    extraction_class: str
    extraction_text: str
    char_interval: Optional[CharInterval] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    alignment_status: str = "aligned"  # aligned, fuzzy, unaligned


@dataclass
class ExampleData:
    """
    Few-Shot 예시 데이터 (LangExtract의 ExampleData 참조)

    LLM에게 추출 패턴을 보여주는 예시입니다.
    """
    text: str
    extractions: List[Extraction] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """추출 결과"""
    text: str
    extractions: List[Extraction] = field(default_factory=list)
    processing_time: float = 0.0
    num_chunks: int = 0
    num_passes: int = 1


# ==================== 한국어 Unicode 토크나이저 ====================

# CJK/Hangul 패턴 (LangExtract의 _CJK_PATTERN 참조) — 사전 컴파일로 성능 최적화
_HANGUL_PATTERN = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]')
_CJK_PATTERN = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]')
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?。？！])\s+')
_WHITESPACE_RE = re.compile(r'\s+')


class KoreanUnicodeTokenizer:
    """
    한국어/CJK 문자를 위한 Unicode 기반 토크나이저

    LangExtract의 UnicodeTokenizer를 참조하여 구현:
    - 한국어 자모/음절 단위 분리
    - CJK 문자 개별 토큰 처리
    - 정확한 char_interval 계산

    참조: https://github.com/google/langextract (Japanese extraction example)
    """

    @staticmethod
    def find_text_positions(
        source_text: str,
        search_text: str,
        fuzzy: bool = True,
    ) -> List[CharInterval]:
        """
        원문에서 검색 텍스트의 정확한 위치를 찾습니다.

        Args:
            source_text: 전체 원문
            search_text: 찾을 텍스트
            fuzzy: 퍼지 매칭 허용 여부

        Returns:
            CharInterval 리스트
        """
        positions = []

        # 1. 정확한 매칭
        start = 0
        while True:
            idx = source_text.find(search_text, start)
            if idx == -1:
                break
            positions.append(CharInterval(
                start_pos=idx,
                end_pos=idx + len(search_text),
            ))
            start = idx + 1

        # 2. 정확한 매칭이 없으면 퍼지 매칭 시도
        if not positions and fuzzy:
            # 공백 정규화 후 재시도 (사전 컴파일된 정규식 사용)
            normalized_source = _WHITESPACE_RE.sub(' ', source_text)
            normalized_search = _WHITESPACE_RE.sub(' ', search_text)

            idx = normalized_source.find(normalized_search)
            if idx != -1:
                # 원문 위치로 역매핑
                orig_idx = _map_normalized_to_original(source_text, idx)
                if orig_idx >= 0:
                    positions.append(CharInterval(
                        start_pos=orig_idx,
                        end_pos=orig_idx + len(search_text),
                    ))

        return positions

    @staticmethod
    def is_hangul(char: str) -> bool:
        """한글 음절/자모 여부 확인"""
        return bool(_HANGUL_PATTERN.match(char))

    @staticmethod
    def count_hangul_ratio(text: str) -> float:
        """텍스트의 한글 비율 계산"""
        if not text:
            return 0.0
        hangul_count = sum(1 for c in text if _HANGUL_PATTERN.match(c))
        return hangul_count / len(text)


def _map_normalized_to_original(original: str, normalized_pos: int) -> int:
    """정규화된 위치를 원문 위치로 매핑"""
    norm_idx = 0
    orig_idx = 0
    in_whitespace = False

    while orig_idx < len(original) and norm_idx < normalized_pos:
        if original[orig_idx].isspace():
            if not in_whitespace:
                norm_idx += 1
                in_whitespace = True
        else:
            norm_idx += 1
            in_whitespace = False
        orig_idx += 1

    return orig_idx


# ==================== 구조화 추출기 ====================

# 기본 한국어 Few-Shot 예시
DEFAULT_KOREAN_EXAMPLES = [
    ExampleData(
        text="삼성전자 이재용 회장이 2025년 반도체 투자 확대 계획을 발표했다.",
        extractions=[
            Extraction(
                extraction_class="조직",
                extraction_text="삼성전자",
                attributes={"산업": "반도체/전자"},
            ),
            Extraction(
                extraction_class="인물",
                extraction_text="이재용",
                attributes={"직함": "회장", "소속": "삼성전자"},
            ),
            Extraction(
                extraction_class="사건",
                extraction_text="반도체 투자 확대 계획을 발표",
                attributes={"시기": "2025년", "분야": "반도체"},
            ),
        ],
    ),
]

EXTRACTION_SYSTEM_PROMPT = """당신은 한국어 문서에서 구조화된 정보를 추출하는 전문가입니다.

### 추출 규칙
{prompt_description}

### 중요 규칙
1. extraction_text는 반드시 원문 텍스트를 그대로 사용 (의역 금지)
2. 등장 순서대로 추출
3. 각 엔티티에 의미 있는 attributes 추가
4. 중복 엔티티는 첫 등장만 추출

### 예시
{examples_text}

### 출력 형식 (JSON)
{{
  "extractions": [
    {{
      "extraction_class": "클래스명",
      "extraction_text": "원문 텍스트",
      "attributes": {{"속성키": "속성값"}},
      "description": "간결한 설명"
    }}
  ]
}}
"""


class StructuredEntityExtractor:
    """
    LangExtract-inspired 구조화된 엔티티 추출기

    Google LangExtract의 핵심 아키텍처를 참조하여 구현:
    - Few-Shot 기반 추출 (사용자 정의 예시로 모델 가이드)
    - 문서 청킹 + 병렬 처리
    - Multi-Pass Extraction (다중 패스로 Recall 향상)
    - Source Grounding (원문 위치 추적)
    - 한국어 Unicode 토크나이저

    참조: https://github.com/google/langextract

    사용 예시:
        extractor = StructuredEntityExtractor(
            prompt_description="문서에서 인물, 조직, 사건을 추출하세요.",
            examples=[...],
        )
        result = extractor.extract("분석할 텍스트...")
    """

    def __init__(
        self,
        prompt_description: str = "문서에서 인물, 조직, 장소, 날짜, 사건, 정책, 금액 등 주요 엔티티를 추출하세요.",
        examples: Optional[List[ExampleData]] = None,
        model_key: str = "gpt-5.4",
        max_chunk_chars: int = 3000,
        extraction_passes: int = 1,
        max_workers: int = 4,
    ):
        self.prompt_description = prompt_description
        self.examples = examples or DEFAULT_KOREAN_EXAMPLES
        self.model_key = model_key
        self.max_chunk_chars = max_chunk_chars
        self.extraction_passes = extraction_passes
        self.max_workers = max_workers

        self.client = AzureClientFactory.get_openai_client(is_advanced=True)
        self.model_name = Config.MODELS.get(model_key, "model-router")
        self._is_gpt5 = "gpt-5" in model_key.lower()
        self.tokenizer = KoreanUnicodeTokenizer()

        # Few-Shot 시스템 프롬프트 사전 구성 (패스마다 재계산 방지)
        examples_text = self._format_examples()
        self._system_prompt = EXTRACTION_SYSTEM_PROMPT.format(
            prompt_description=self.prompt_description,
            examples_text=examples_text,
        )

        print(f"📋 StructuredEntityExtractor 초기화 "
              f"(모델: {model_key}, 패스: {extraction_passes}, 청크: {max_chunk_chars}자)")

    def extract(
        self,
        text: str,
        additional_context: str = "",
    ) -> ExtractionResult:
        """
        텍스트에서 구조화된 엔티티를 추출합니다.

        Args:
            text: 추출 대상 텍스트
            additional_context: 추가 문맥 정보

        Returns:
            ExtractionResult: 추출 결과
        """
        start_time = time.time()

        # 1. 텍스트 청킹
        chunks = self._chunk_text(text)
        print(f"   📄 텍스트 청킹 완료: {len(chunks)}개 청크")

        # 2. Multi-Pass Extraction
        all_extractions = []
        for pass_num in range(1, self.extraction_passes + 1):
            if self.extraction_passes > 1:
                print(f"   🔄 추출 패스 {pass_num}/{self.extraction_passes}")

            pass_extractions = self._extract_from_chunks(
                chunks, additional_context, pass_num
            )
            all_extractions.append(pass_extractions)

        # 3. 패스간 중복 제거 및 병합
        merged = self._merge_extractions(all_extractions)

        # 4. Source Grounding (원문 위치 추적)
        self._ground_extractions(text, merged)

        elapsed = time.time() - start_time

        print(f"   ✅ 추출 완료: {len(merged)}개 엔티티 ({elapsed:.1f}초)")

        return ExtractionResult(
            text=text,
            extractions=merged,
            processing_time=elapsed,
            num_chunks=len(chunks),
            num_passes=self.extraction_passes,
        )

    def extract_from_document_chunks(
        self,
        chunks: List[Any],
        additional_context: str = "",
    ) -> ExtractionResult:
        """
        이미 청킹된 문서 청크에서 엔티티를 추출합니다.

        Args:
            chunks: Document 객체 리스트 (page_content 속성 필요)
            additional_context: 추가 문맥

        Returns:
            ExtractionResult
        """
        start_time = time.time()

        # Document 객체에서 텍스트 추출
        text_chunks = []
        for chunk in chunks:
            if hasattr(chunk, "page_content"):
                text_chunks.append(chunk.page_content)
            elif isinstance(chunk, dict):
                text_chunks.append(chunk.get("page_content", str(chunk)))
            else:
                text_chunks.append(str(chunk))

        full_text = "\n\n".join(text_chunks)

        all_extractions = []
        for pass_num in range(1, self.extraction_passes + 1):
            pass_extractions = self._extract_from_chunks(
                text_chunks, additional_context, pass_num
            )
            all_extractions.append(pass_extractions)

        merged = self._merge_extractions(all_extractions)
        self._ground_extractions(full_text, merged)

        elapsed = time.time() - start_time

        return ExtractionResult(
            text=full_text,
            extractions=merged,
            processing_time=elapsed,
            num_chunks=len(text_chunks),
            num_passes=self.extraction_passes,
        )

    def _chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할 (LangExtract의 ChunkIterator 참조)

        한국어 문장 경계를 존중하여 분할합니다.
        """
        if len(text) <= self.max_chunk_chars:
            return [text]

        chunks = []
        # 단락 단위로 먼저 분할 (사전 컴파일된 정규식 사용)
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_chars:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 단락 자체가 너무 길면 문장 단위로 분할
                if len(para) > self.max_chunk_chars:
                    sentences = _SENTENCE_SPLIT_RE.split(para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.max_chunk_chars:
                            current_chunk += (" " + sent if current_chunk else sent)
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _extract_from_chunks(
        self,
        chunks: List[str],
        additional_context: str,
        pass_num: int,
    ) -> List[Extraction]:
        """청크들에서 병렬로 엔티티 추출"""
        all_extractions = []

        # 병렬 처리
        if len(chunks) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._extract_single_chunk,
                        chunk,
                        self._system_prompt,
                        additional_context,
                    ): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    try:
                        extractions = future.result()
                        all_extractions.extend(extractions)
                    except Exception as e:
                        print(f"      ⚠️ 청크 추출 실패: {e}")
        else:
            for chunk in chunks:
                extractions = self._extract_single_chunk(
                    chunk, self._system_prompt, additional_context
                )
                all_extractions.extend(extractions)

        return all_extractions

    def _extract_single_chunk(
        self,
        chunk_text: str,
        system_prompt: str,
        additional_context: str,
    ) -> List[Extraction]:
        """단일 청크에서 엔티티 추출"""
        user_message = f"다음 텍스트에서 엔티티를 추출하세요:\n\n{chunk_text}"
        if additional_context:
            user_message += f"\n\n추가 문맥: {additional_context}"

        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            if self._is_gpt5:
                completion_params["max_completion_tokens"] = 4000
            else:
                completion_params["max_tokens"] = 4000

            response = self.client.chat.completions.create(**completion_params)
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            extractions = []
            for ext_data in result.get("extractions", []):
                extraction = Extraction(
                    extraction_class=ext_data.get("extraction_class", "기타"),
                    extraction_text=ext_data.get("extraction_text", ""),
                    attributes=ext_data.get("attributes", {}),
                    description=ext_data.get("description", ""),
                )
                if extraction.extraction_text:  # 빈 텍스트 제외
                    extractions.append(extraction)

            return extractions

        except Exception as e:
            print(f"      ⚠️ LLM 추출 오류: {e}")
            return []

    def _format_examples(self) -> str:
        """Few-Shot 예시를 프롬프트 형식으로 변환"""
        parts = []
        for i, example in enumerate(self.examples, 1):
            parts.append(f"예시 {i}:")
            parts.append(f"텍스트: {example.text}")
            for ext in example.extractions:
                attrs = ", ".join(f'{k}: {v}' for k, v in ext.attributes.items())
                parts.append(
                    f"  → [{ext.extraction_class}] \"{ext.extraction_text}\""
                    + (f" ({attrs})" if attrs else "")
                )
            parts.append("")
        return "\n".join(parts)

    def _merge_extractions(
        self,
        all_passes: List[List[Extraction]],
    ) -> List[Extraction]:
        """
        Multi-Pass 추출 결과 병합 (LangExtract의 _merge_non_overlapping_extractions 참조)

        중복 엔티티를 제거하고, 가장 완전한 결과를 유지합니다.
        """
        if len(all_passes) == 1:
            return self._deduplicate(all_passes[0])

        # 모든 패스의 결과를 합침
        combined = []
        for pass_extractions in all_passes:
            combined.extend(pass_extractions)

        return self._deduplicate(combined)

    def _deduplicate(self, extractions: List[Extraction]) -> List[Extraction]:
        """추출 결과 중복 제거"""
        seen = set()
        unique = []

        for ext in extractions:
            # (클래스, 텍스트) 기준으로 중복 체크
            key = (ext.extraction_class, ext.extraction_text.strip())
            if key not in seen:
                seen.add(key)
                unique.append(ext)

        return unique

    def _ground_extractions(
        self,
        full_text: str,
        extractions: List[Extraction],
    ) -> None:
        """
        Source Grounding: 추출된 엔티티의 원문 위치를 추적합니다.
        (LangExtract의 Resolver.align 참조)
        """
        for ext in extractions:
            positions = self.tokenizer.find_text_positions(
                full_text, ext.extraction_text, fuzzy=True
            )
            if positions:
                ext.char_interval = positions[0]  # 첫 등장 위치
                ext.alignment_status = "aligned"
            else:
                ext.alignment_status = "unaligned"

    def extractions_to_dict(
        self,
        result: ExtractionResult,
    ) -> List[Dict[str, Any]]:
        """추출 결과를 직렬화 가능한 딕셔너리로 변환"""
        return [
            {
                "extraction_class": e.extraction_class,
                "extraction_text": e.extraction_text,
                "char_interval": {
                    "start": e.char_interval.start_pos,
                    "end": e.char_interval.end_pos,
                } if e.char_interval else None,
                "attributes": e.attributes,
                "description": e.description,
                "alignment_status": e.alignment_status,
            }
            for e in result.extractions
        ]
