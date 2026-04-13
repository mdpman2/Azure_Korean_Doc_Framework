"""
에러 자동 복구 및 재시도 시스템.

Claude Code / OpenClaude의 에러 보류(Error Withholding) + 재시도 전략을 참조.
- 429 (Rate Limit): 지수 백오프 재시도
- 413 (컨텍스트 초과): 자동 컨텍스트 축소 후 재시도
- 500/503 (서버 에러): 폴백 모델 전환
- 일반 에러: 최대 N회 재시도

[v5.0 신규]
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


class ErrorClass(str, Enum):
    """분류된 에러 유형."""
    RATE_LIMIT = "rate_limit"           # 429
    CONTEXT_OVERFLOW = "context_overflow"  # 413 / context_length_exceeded
    SERVER_ERROR = "server_error"       # 500, 502, 503
    AUTH_ERROR = "auth_error"           # 401, 403
    TIMEOUT = "timeout"                 # 연결/읽기 타임아웃
    MODEL_UNAVAILABLE = "model_unavailable"  # 모델 과부하/비가용
    UNKNOWN = "unknown"


@dataclass
class RetryRecord:
    """재시도 기록."""
    attempt: int
    error_class: ErrorClass
    error_message: str
    wait_seconds: float
    model_key: Optional[str] = None
    action: str = ""   # "retry", "fallback", "compact", "fail"


@dataclass
class RecoveryResult:
    """복구 시도 결과."""
    success: bool
    result: Any = None
    total_attempts: int = 0
    retry_records: List[RetryRecord] = field(default_factory=list)
    final_error: Optional[str] = None
    final_model: Optional[str] = None


def classify_error(exc: Exception) -> ErrorClass:
    """예외를 에러 클래스로 분류합니다."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__

    # openai 라이브러리의 구체적 예외 타입 확인
    if "ratelimiterror" in exc_type.lower() or "429" in exc_str or "rate limit" in exc_str:
        return ErrorClass.RATE_LIMIT
    if "context_length_exceeded" in exc_str or "413" in exc_str or "too many tokens" in exc_str or "maximum context" in exc_str:
        return ErrorClass.CONTEXT_OVERFLOW
    if "autherror" in exc_type.lower() or "401" in exc_str or "403" in exc_str or "unauthorized" in exc_str:
        return ErrorClass.AUTH_ERROR
    if "timeout" in exc_type.lower() or "timed out" in exc_str or "timeout" in exc_str:
        return ErrorClass.TIMEOUT
    if any(code in exc_str for code in ("500", "502", "503", "overloaded", "capacity")):
        return ErrorClass.SERVER_ERROR
    if "model" in exc_str and ("not found" in exc_str or "unavailable" in exc_str or "does not exist" in exc_str):
        return ErrorClass.MODEL_UNAVAILABLE

    return ErrorClass.UNKNOWN


class RetryPolicy:
    """재시도 정책 설정."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        fallback_models: Optional[List[str]] = None,
        compact_on_overflow: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.fallback_models = fallback_models or ["gpt-5.2", "gpt-4.1"]
        self.compact_on_overflow = compact_on_overflow

    def get_delay(self, attempt: int, error_class: ErrorClass) -> float:
        """지수 백오프 + 지터로 대기 시간을 계산합니다."""
        if error_class == ErrorClass.RATE_LIMIT:
            delay = self.base_delay * (2 ** attempt)
        elif error_class == ErrorClass.SERVER_ERROR:
            delay = self.base_delay * (1.5 ** attempt)
        else:
            delay = self.base_delay * attempt

        delay = min(delay, self.max_delay)
        if self.jitter:
            delay += random.uniform(0, delay * 0.25)
        return delay


class ErrorRecoveryManager:
    """
    에러 자동 복구 매니저.

    사용 예시:
        recovery = ErrorRecoveryManager()
        result = recovery.execute_with_retry(
            fn=lambda model_key: model_manager.get_completion("hello", model_key=model_key),
            model_key="gpt-5.4",
        )
        if result.success:
            print(result.result)
        else:
            print(f"Failed after {result.total_attempts} attempts: {result.final_error}")
    """

    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()

    def execute_with_retry(
        self,
        fn: Callable[..., T],
        model_key: Optional[str] = None,
        compact_fn: Optional[Callable[[], None]] = None,
        **fn_kwargs,
    ) -> RecoveryResult:
        """
        함수를 재시도 정책에 따라 실행합니다.

        Args:
            fn: 실행할 함수. 키워드 인자로 model_key를 받을 수 있음.
            model_key: 초기 모델 키.
            compact_fn: 컨텍스트 초과 시 실행할 축소 함수.
            **fn_kwargs: fn에 전달할 추가 키워드 인자.
        """
        records: List[RetryRecord] = []
        current_model = model_key
        fallback_index = 0

        for attempt in range(self.policy.max_retries + 1):
            try:
                kwargs = dict(fn_kwargs)
                if current_model is not None:
                    kwargs["model_key"] = current_model
                result = fn(**kwargs)
                return RecoveryResult(
                    success=True,
                    result=result,
                    total_attempts=attempt + 1,
                    retry_records=records,
                    final_model=current_model,
                )

            except Exception as exc:
                error_class = classify_error(exc)
                wait = self.policy.get_delay(attempt, error_class)
                action = "retry"

                # Auth 에러는 재시도 불가
                if error_class == ErrorClass.AUTH_ERROR:
                    records.append(RetryRecord(
                        attempt=attempt, error_class=error_class,
                        error_message=str(exc), wait_seconds=0,
                        model_key=current_model, action="fail",
                    ))
                    return RecoveryResult(
                        success=False, total_attempts=attempt + 1,
                        retry_records=records, final_error=str(exc),
                        final_model=current_model,
                    )

                # 컨텍스트 오버플로우: 축소 시도
                if error_class == ErrorClass.CONTEXT_OVERFLOW and self.policy.compact_on_overflow:
                    if compact_fn is not None:
                        try:
                            compact_fn()
                            action = "compact"
                            wait = 0.5
                            print(f"   🗜️ 컨텍스트 축소 후 재시도... (attempt {attempt + 1})")
                        except Exception:
                            action = "fallback"

                # 서버 에러 / 모델 비가용: 폴백 모델 전환
                if error_class in (ErrorClass.SERVER_ERROR, ErrorClass.MODEL_UNAVAILABLE):
                    if fallback_index < len(self.policy.fallback_models):
                        current_model = self.policy.fallback_models[fallback_index]
                        fallback_index += 1
                        action = "fallback"
                        print(f"   🔄 폴백 모델 전환: {current_model} (attempt {attempt + 1})")

                records.append(RetryRecord(
                    attempt=attempt, error_class=error_class,
                    error_message=str(exc), wait_seconds=wait,
                    model_key=current_model, action=action,
                ))

                if attempt < self.policy.max_retries:
                    print(f"   ⏳ {error_class.value} 에러, {wait:.1f}s 후 재시도 ({attempt + 1}/{self.policy.max_retries})")
                    time.sleep(wait)

        return RecoveryResult(
            success=False,
            total_attempts=self.policy.max_retries + 1,
            retry_records=records,
            final_error=records[-1].error_message if records else "Unknown error",
            final_model=current_model,
        )
