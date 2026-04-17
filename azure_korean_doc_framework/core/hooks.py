"""
Hook 시스템 — 파이프라인 각 단계에 사용자 정의 콜백을 등록/실행하는 시스템.

Claude Code 소스 분석의 PreToolUse / PostToolUse 개념을 RAG 파이프라인에 적용.
검색 전/후, 답변 생성 전/후, 가드레일 전/후 훅 포인트를 제공합니다.

[v5.0 신규]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HookEvent(str, Enum):
    """등록 가능한 훅 이벤트."""
    PRE_SEARCH = "pre_search"
    POST_SEARCH = "post_search"
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    PRE_GUARDRAIL = "pre_guardrail"
    POST_GUARDRAIL = "post_guardrail"
    PRE_EVIDENCE = "pre_evidence"
    POST_EVIDENCE = "post_evidence"
    ON_ERROR = "on_error"
    ON_STREAM_TOKEN = "on_stream_token"


@dataclass
class HookContext:
    """훅 콜백이 받는 컨텍스트 객체."""
    event: HookEvent
    data: Dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True
    modified_data: Optional[Dict[str, Any]] = None

    def block(self):
        """파이프라인 진행을 차단합니다."""
        self.should_continue = False

    def modify(self, **kwargs):
        """데이터를 수정합니다."""
        if self.modified_data is None:
            self.modified_data = {}
        self.modified_data.update(kwargs)


@dataclass
class HookResult:
    """훅 실행 결과."""
    event: HookEvent
    hook_count: int
    blocked: bool = False
    modified_data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


# 훅 콜백 시그니처: (HookContext) -> None
HookCallback = Callable[[HookContext], None]


class HookRegistry:
    """
    파이프라인 훅 레지스트리.

    사용 예시:
        registry = HookRegistry()

        # 검색 전에 쿼리 로그
        @registry.on(HookEvent.PRE_SEARCH)
        def log_query(ctx: HookContext):
            print(f"Searching: {ctx.data.get('question')}")

        # 답변 후 커스텀 후처리
        @registry.on(HookEvent.POST_GENERATION)
        def postprocess(ctx: HookContext):
            answer = ctx.data.get('answer', '')
            ctx.modify(answer=answer + '\\n---\\n후처리 완료')

        # 에이전트에 등록
        agent = KoreanDocAgent()
        agent.hook_registry = registry
    """

    def __init__(self):
        self._hooks: Dict[HookEvent, List[HookCallback]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, callback: HookCallback, *, priority: int = 0):
        """이벤트에 콜백을 등록합니다."""
        self._hooks[event].append((priority, callback))
        self._hooks[event].sort(key=lambda x: x[0], reverse=True)

    def on(self, event: HookEvent, *, priority: int = 0):
        """데코레이터로 훅을 등록합니다."""
        def decorator(fn: HookCallback) -> HookCallback:
            self.register(event, fn, priority=priority)
            return fn
        return decorator

    def unregister(self, event: HookEvent, callback: HookCallback):
        """이벤트에서 특정 콜백을 제거합니다."""
        self._hooks[event] = [(p, cb) for p, cb in self._hooks[event] if cb is not callback]

    def clear(self, event: Optional[HookEvent] = None):
        """이벤트(또는 전체)의 훅을 제거합니다."""
        if event:
            self._hooks[event] = []
        else:
            for ev in HookEvent:
                self._hooks[ev] = []

    def run(self, event: HookEvent, data: Optional[Dict[str, Any]] = None) -> HookResult:
        """등록된 훅을 실행하고 결과를 반환합니다."""
        hooks = list(self._hooks.get(event, []))  # 스냅샷: 이터레이션 중 변경 방지
        ctx = HookContext(event=event, data=data or {})
        errors: List[str] = []
        start = time.perf_counter()

        for _priority, callback in hooks:
            try:
                callback(ctx)
                if not ctx.should_continue:
                    break
            except Exception as exc:
                errors.append(f"{callback.__name__}: {exc}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        return HookResult(
            event=event,
            hook_count=len(hooks),
            blocked=not ctx.should_continue,
            modified_data=ctx.modified_data,
            errors=errors,
            elapsed_ms=elapsed_ms,
        )

    @property
    def registered_count(self) -> Dict[HookEvent, int]:
        return {event: len(hooks) for event, hooks in self._hooks.items()}
