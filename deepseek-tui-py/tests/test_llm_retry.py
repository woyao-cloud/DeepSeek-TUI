"""Tests for the LLM retry logic."""

import pytest

from deepseek_tui.llm import RetryConfig, with_retry, LlmError, RetryError


class TestLlmError:
    def test_retryable_errors(self):
        assert LlmError.rate_limited().is_retryable()
        assert LlmError.server_error(500).is_retryable()
        assert LlmError.network("timeout").is_retryable()
        assert LlmError.timeout(30).is_retryable()

    def test_non_retryable_errors(self):
        assert not LlmError.auth("invalid key").is_retryable()
        assert not LlmError.invalid_request(400).is_retryable()
        assert not LlmError.context_length("too long").is_retryable()
        assert not LlmError.content_policy("unsafe").is_retryable()

    def test_from_http_response(self):
        err = LlmError.from_http_response(429, "rate limited")
        assert err.kind == "rate_limited"
        assert err.is_retryable()

        err = LlmError.from_http_response(401, "unauthorized")
        assert err.kind == "authentication_error"
        assert not err.is_retryable()

        err = LlmError.from_http_response(500, "server error")
        assert err.kind == "server_error"
        assert err.is_retryable()

        err = LlmError.from_http_response(400, "context_length_exceeded")
        assert err.kind == "context_length_error"
        assert not err.is_retryable()

    def test_string_representation(self):
        err = LlmError.rate_limited("too many requests")
        assert "rate_limited" in str(err)


class TestRetryConfig:
    def test_delay_exponential(self):
        config = RetryConfig()
        d0 = config.initial_delay * (config.exponential_base ** 0)
        d1 = config.initial_delay * (config.exponential_base ** 1)
        d2 = config.initial_delay * (config.exponential_base ** 2)
        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0


class TestWithRetry:
    async def test_success_first_attempt(self):
        config = RetryConfig()
        result = await with_retry(config, lambda: _ok(42))
        assert result == 42

    async def test_retry_disabled(self):
        config = RetryConfig(enabled=False)
        with pytest.raises(LlmError):
            await with_retry(config, lambda: _fail_recoverable())

    async def test_non_retryable_fails_immediately(self):
        config = RetryConfig()
        with pytest.raises(RetryError):
            await with_retry(config, lambda: _fail_auth())

    async def test_eventual_success(self):
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LlmError.server_error(500, "temp")
            return 42

        result = await with_retry(config, flaky_op)
        assert result == 42
        assert call_count == 3


async def _ok(value):
    return value


async def _fail_recoverable():
    raise LlmError.server_error(500, "error")


async def _fail_auth():
    raise LlmError.auth("invalid key")
