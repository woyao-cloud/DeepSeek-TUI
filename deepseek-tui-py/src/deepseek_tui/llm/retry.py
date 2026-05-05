"""Retry logic for LLM API calls — port of `crates/tui/src/llm_client/`.

Provides RetryConfig, LlmError classification, and with_retry wrapper.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


# ── LlmError — Classified Error Types ─────────────────────────────────

class LlmError(Exception):
    """Classified LLM errors with retryability information."""

    def __init__(
        self,
        kind: str,
        message: str = "",
        retryable: bool = False,
        retry_after: Optional[float] = None,
        status: Optional[int] = None,
    ) -> None:
        self.kind = kind
        self.message = message
        self._retryable = retryable
        self.retry_after = retry_after
        self.status = status
        super().__init__(self._format())

    def _format(self) -> str:
        if self.status:
            return f"{self.kind} ({self.status}): {self.message}"
        return f"{self.kind}: {self.message}"

    @classmethod
    def rate_limited(cls, message: str = "", retry_after: Optional[float] = None) -> LlmError:
        return cls("rate_limited", message, retryable=True, retry_after=retry_after, status=429)

    @classmethod
    def server_error(cls, status: int, message: str = "") -> LlmError:
        return cls("server_error", message, retryable=True, status=status)

    @classmethod
    def network(cls, message: str = "") -> LlmError:
        return cls("network_error", message, retryable=True)

    @classmethod
    def timeout(cls, seconds: float = 0) -> LlmError:
        return cls("timeout", f"timed out after {seconds}s", retryable=True)

    @classmethod
    def auth(cls, message: str = "") -> LlmError:
        return cls("authentication_error", message, retryable=False, status=401)

    @classmethod
    def invalid_request(cls, status: int, message: str = "") -> LlmError:
        return cls("invalid_request", message, retryable=False, status=status)

    @classmethod
    def context_length(cls, message: str = "") -> LlmError:
        return cls("context_length_error", message, retryable=False)

    @classmethod
    def content_policy(cls, message: str = "") -> LlmError:
        return cls("content_policy_error", message, retryable=False)

    @classmethod
    def model_error(cls, message: str = "") -> LlmError:
        return cls("model_error", message, retryable=False)

    @classmethod
    def parse(cls, message: str = "") -> LlmError:
        return cls("parse_error", message, retryable=False)

    @classmethod
    def from_http_response(cls, status: int, body: str) -> LlmError:
        if status == 429:
            return cls.rate_limited(body)
        if status in (401, 403):
            return cls.auth(body)
        if status == 400:
            lower = body.lower()
            if any(k in lower for k in ("context_length", "token", "too long", "maximum")):
                return cls.context_length(body)
            if any(k in lower for k in ("content_policy", "safety", "harmful", "inappropriate")):
                return cls.content_policy(body)
            if "model" in lower and "not found" in lower:
                return cls.model_error(body)
            return cls.invalid_request(status, body)
        if status == 404:
            return cls.model_error(body) if "model" in body.lower() else cls.invalid_request(status, body)
        if 500 <= status <= 599:
            return cls.server_error(status, body)
        return cls("unknown", f"HTTP {status}: {body}", retryable=False)

    def is_retryable(self) -> bool:
        return self._retryable


# ── RetryConfig ───────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    respect_retry_after: bool = True


# ── RetryError ────────────────────────────────────────────────────────

class RetryError(Exception):
    """Error returned when all retry attempts have been exhausted."""

    def __init__(self, last_error: LlmError, attempts: int, total_time: float) -> None:
        self.last_error = last_error
        self.attempts = attempts
        self.total_time = total_time
        super().__init__(f"Retry exhausted after {attempts} attempts ({total_time:.1f}s): {last_error}")


# ── with_retry ────────────────────────────────────────────────────────

async def with_retry(
    config: RetryConfig,
    operation: Callable[[], Awaitable[T]],
    callback: Optional[Callable[[LlmError, int, float], None]] = None,
) -> T:
    """Execute an async operation with configurable retry logic."""
    if not config.enabled:
        return await operation()

    start = time.monotonic()
    last_error: Optional[LlmError] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await operation()
        except LlmError as err:
            if not err.is_retryable():
                raise RetryError(err, attempt + 1, time.monotonic() - start) from err

            if attempt >= config.max_retries:
                raise RetryError(err, attempt + 1, time.monotonic() - start) from err

            delay = config.initial_delay * (config.exponential_base ** attempt)
            delay = min(delay, config.max_delay)

            if config.respect_retry_after and err.retry_after is not None:
                delay = min(err.retry_after, config.max_delay)

            if config.jitter:
                jitter_range = delay * config.jitter_factor
                delay += random.uniform(-jitter_range, jitter_range)
                delay = max(0.0, delay)

            if callback:
                callback(err, attempt, delay)

            await asyncio.sleep(delay)
            last_error = err

    raise RetryError(
        last_error or LlmError("unknown", "unknown error"),
        config.max_retries + 1,
        time.monotonic() - start,
    )
