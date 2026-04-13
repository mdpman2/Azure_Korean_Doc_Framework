"""
스트리밍 응답 및 자동 압축 시스템.

Claude Code 소스 분석의 스트리밍 도구 실행기 + 자동 압축(Auto-Compact) 개념을 참조.
- 토큰 단위 실시간 스트리밍 출력
- 대화 히스토리가 토큰 한계에 근접하면 LLM으로 핵심 요약 후 압축

[v5.0 신규]
"""

from __future__ import annotations

import tiktoken
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

from ..config import Config
from ..utils.azure_clients import AzureClientFactory


@dataclass
class StreamChunk:
    """스트리밍 청크."""
    text: str
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactResult:
    """압축 결과."""
    original_token_count: int
    compacted_token_count: int
    summary: str
    removed_message_count: int


class StreamingManager:
    """
    Azure OpenAI 스트리밍 응답 관리자.

    사용 예시:
        manager = StreamingManager()
        for chunk in manager.stream_completion("질문에 답하세요", system_msg="전문가"):
            print(chunk.text, end="", flush=True)
    """

    def __init__(self, model_key: Optional[str] = None):
        self.model_key = model_key or Config.DEFAULT_MODEL
        self._on_token: Optional[Callable[[str], None]] = None

    def on_token(self, callback: Callable[[str], None]):
        """토큰 수신 시 호출되는 콜백을 등록합니다."""
        self._on_token = callback

    def stream_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        model_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> Generator[StreamChunk, None, None]:
        """
        스트리밍으로 LLM 응답을 생성합니다.

        Yields:
            StreamChunk: 텍스트 청크
        """
        key = model_key or self.model_key
        model_name = Config.MODELS.get(key, key)

        from functools import lru_cache
        is_advanced = key in Config.ADVANCED_MODELS
        is_gpt5 = key.startswith("gpt-5") or key.startswith("o3") or key.startswith("o4")

        client = AzureClientFactory.get_openai_client(is_advanced=is_advanced)

        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": True,
        }
        if is_gpt5:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        try:
            response = client.chat.completions.create(**params)
            full_text = ""
            for event in response:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta and delta.content:
                    text = delta.content
                    full_text += text
                    if self._on_token:
                        self._on_token(text)
                    yield StreamChunk(text=text, is_final=False)

            yield StreamChunk(text="", is_final=True, metadata={"full_text": full_text})

        except Exception as e:
            yield StreamChunk(text=f"\n❌ 스트리밍 오류: {e}", is_final=True, metadata={"error": str(e)})

    def stream_rag_answer(
        self,
        question: str,
        contexts: List[str],
        system_prompt: str = "",
        model_key: Optional[str] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        RAG 답변을 스트리밍으로 생성합니다.

        Args:
            question: 사용자 질문
            contexts: 검색된 문맥 리스트
            system_prompt: 시스템 프롬프트
            model_key: 모델 키
        """
        context_str = "\n\n".join(contexts) if contexts else "관련된 문서 내용을 찾을 수 없습니다."
        user_prompt = f"[Context]\n{context_str}\n\n[Question]\n{question}"

        if not system_prompt:
            system_prompt = (
                "당신은 문서 분석 및 Q&A 전문가입니다. "
                "주어진 [Context] 내용을 바탕으로 사용자의 [Question]에 한국어로 친절하고 정확하게 답변하세요."
            )

        yield from self.stream_completion(
            prompt=user_prompt,
            system_message=system_prompt,
            model_key=model_key,
        )


class ContextCompactor:
    """
    대화 컨텍스트 자동 압축 관리자.

    대화 히스토리가 토큰 한계에 근접하면 LLM으로 핵심을 요약하여 압축합니다.
    Claude Code의 Auto-Compact 서비스를 참조.

    사용 예시:
        compactor = ContextCompactor(max_context_tokens=120000)

        # 검색 컨텍스트가 너무 길 때 압축
        if compactor.should_compact(contexts):
            compacted = compactor.compact_contexts(contexts, question="인사제도 담당자는?")
            contexts = [compacted.summary]
    """

    def __init__(
        self,
        max_context_tokens: int = 120000,
        compact_threshold_ratio: float = 0.85,
        model_key: Optional[str] = None,
    ):
        self.max_context_tokens = max_context_tokens
        self.compact_threshold = int(max_context_tokens * compact_threshold_ratio)
        self.model_key = model_key or Config.DEFAULT_MODEL
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산합니다."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # 폴백: 대략적 추정 (한국어 평균 1.5 토큰/글자)
        return int(len(text) * 1.5)

    def count_context_tokens(self, contexts: List[str]) -> int:
        """컨텍스트 리스트의 총 토큰 수."""
        return sum(self.count_tokens(c) for c in contexts)

    def should_compact(self, contexts: List[str]) -> bool:
        """압축이 필요한지 판단합니다."""
        return self.count_context_tokens(contexts) > self.compact_threshold

    def compact_contexts(
        self,
        contexts: List[str],
        question: str = "",
        max_summary_tokens: int = 2000,
    ) -> CompactResult:
        """
        검색된 컨텍스트를 LLM으로 요약/압축합니다.

        선택 전략:
        1. 토큰 수 기준으로 가장 긴 컨텍스트부터 요약
        2. 질문과 관련된 핵심 정보만 보존
        3. 중복 정보 제거
        """
        from ..core.multi_model_manager import MultiModelManager

        original_count = self.count_context_tokens(contexts)
        if not self.should_compact(contexts):
            return CompactResult(
                original_token_count=original_count,
                compacted_token_count=original_count,
                summary="\n\n".join(contexts),
                removed_message_count=0,
            )

        # 토큰 예산 내에서 가능한 많은 컨텍스트 유지
        kept = []
        kept_tokens = 0
        to_summarize = []

        # 짧은 컨텍스트부터 유지 (핵심 정보가 짧은 경향)
        sorted_by_len = sorted(enumerate(contexts), key=lambda x: len(x[1]))

        budget = self.compact_threshold // 2  # 절반은 원본 유지, 절반은 요약

        for idx, ctx in sorted_by_len:
            ctx_tokens = self.count_tokens(ctx)
            if kept_tokens + ctx_tokens <= budget:
                kept.append((idx, ctx))
                kept_tokens += ctx_tokens
            else:
                to_summarize.append(ctx)

        if not to_summarize:
            return CompactResult(
                original_token_count=original_count,
                compacted_token_count=kept_tokens,
                summary="\n\n".join(c for _, c in sorted(kept)),
                removed_message_count=0,
            )

        # 요약 대상 텍스트를 LLM으로 압축
        summary_input = "\n\n---\n\n".join(to_summarize)
        question_hint = f"\n관련 질문: {question}" if question else ""

        summary_prompt = (
            f"다음 문서 검색 결과를 핵심 정보만 보존하여 간결하게 요약하세요.{question_hint}\n"
            f"중복 제거, 수치/이름/날짜 정확히 보존, 출처 정보 유지.\n\n"
            f"검색 결과:\n{summary_input}"
        )

        model_manager = MultiModelManager(default_model=self.model_key)
        summary = model_manager.get_completion(
            prompt=summary_prompt,
            system_message="검색 결과를 정밀하게 요약하는 전문가입니다. 핵심 사실만 보존하세요.",
            temperature=0.0,
            max_tokens=max_summary_tokens,
        )

        # 유지된 컨텍스트 + 요약 결합
        final_parts = [c for _, c in sorted(kept)]
        final_parts.append(f"[요약된 검색 결과]\n{summary}")
        final_text = "\n\n".join(final_parts)

        return CompactResult(
            original_token_count=original_count,
            compacted_token_count=self.count_tokens(final_text),
            summary=final_text,
            removed_message_count=len(to_summarize),
        )
