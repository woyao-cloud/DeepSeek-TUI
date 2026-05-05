"""LLM Client abstraction and DeepSeek HTTP client — port of `crates/tui/src/client/` and `llm_client/`.

Provides the `LlmClient` ABC and the `DeepSeekClient` implementation with streaming SSE support.
"""

from __future__ import annotations

import abc
import json
from typing import Any, AsyncIterator, Optional

import httpx
from httpx_sse import EventSource, aconnect_sse

from .models import MessageRequest, MessageResponse, StreamEvent
from .retry import LlmError


class LlmClient(abc.ABC):
    """Unified interface for LLM providers."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def model(self) -> str:
        ...

    @abc.abstractmethod
    async def create_message(self, request: MessageRequest) -> MessageResponse:
        ...

    @abc.abstractmethod
    async def create_message_stream(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamEvent]:
        ...

    async def health_check(self) -> bool:
        return True


_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_DEFAULT_TIMEOUT = 120.0


class DeepSeekClient(LlmClient):
    """HTTP client for DeepSeek's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-pro",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    async def close(self) -> None:
        await self._client.aclose()

    # ── Non-streaming ──────────────────────────────────────────────

    async def create_message(self, request: MessageRequest) -> MessageResponse:
        payload = self._build_payload(request, stream=False)
        try:
            response = await self._client.post(_CHAT_COMPLETIONS_PATH, json=payload)
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except httpx.HTTPStatusError as e:
            raise _http_error_to_llm(e) from e
        except httpx.RequestError as e:
            raise LlmError.network(f"Request failed: {e}") from e

    # ── Streaming ──────────────────────────────────────────────────

    async def create_message_stream(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        try:
            async with aconnect_sse(
                self._client, "POST", _CHAT_COMPLETIONS_PATH, json=payload
            ) as event_source:
                async for event in event_source.aiter_sse():
                    if event.event == "ping":
                        continue
                    if event.data == "[DONE]":
                        break
                    try:
                        data = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
                    yield self._parse_stream_event(data)
        except httpx.HTTPStatusError as e:
            raise _http_error_to_llm(e) from e
        except httpx.RequestError as e:
            raise LlmError.network(f"Stream failed: {e}") from e

    # ── Payload Building ───────────────────────────────────────────

    def _build_payload(self, request: MessageRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": self._serialize_messages(request.messages),
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.tools is not None:
            payload["tools"] = self._serialize_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.thinking is not None:
            payload["thinking"] = request.thinking
        if request.system is not None:
            payload["system"] = self._serialize_system(request.system)
        return payload

    @staticmethod
    def _serialize_messages(messages: list) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role}
            if msg.content:
                entry["content"] = [
                    _serialize_content_block(cb) for cb in msg.content
                ]
            result.append(entry)
        return result

    @staticmethod
    def _serialize_system(system) -> Any:
        if system.text:
            return system.text
        if system.blocks:
            return [
                {
                    "type": b.type,
                    "text": b.text,
                    **({"cache_control": {"type": b.cache_control.type}} if b.cache_control else {}),
                }
                for b in system.blocks
            ]
        return None

    @staticmethod
    def _serialize_tools(tools: list) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    # ── Response Parsing ───────────────────────────────────────────

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> MessageResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content_blocks = []
        content_raw = message.get("content")
        if content_raw:
            content_blocks.append({
                "type": "text",
                "text": content_raw,
            })
        for tool_call in message.get("tool_calls") or []:
            content_blocks.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": tool_call["function"]["name"],
                "input": json.loads(tool_call["function"].get("arguments", "{}")),
            })
        usage_data = data.get("usage", {})
        return MessageResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content_blocks,
            stop_reason=choice.get("finish_reason"),
            usage={
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            },
        )

    @staticmethod
    def _parse_stream_event(data: dict[str, Any]) -> StreamEvent:
        event_type = data.get("type", "")
        if event_type == "message_start":
            msg = data.get("message", {})
            return StreamEvent(
                type="message_start",
                message=DeepSeekClient._parse_response(msg),
            )
        if event_type == "content_block_start":
            return StreamEvent(
                type="content_block_start",
                index=data.get("index"),
                content_block=data.get("content_block"),
            )
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            return StreamEvent(
                type="content_block_delta",
                index=data.get("index"),
                delta=delta,
            )
        if event_type == "content_block_stop":
            return StreamEvent(type="content_block_stop", index=data.get("index"))
        if event_type == "message_delta":
            return StreamEvent(
                type="message_delta",
                delta=data.get("delta"),
                usage=data.get("usage"),
            )
        if event_type == "message_stop":
            return StreamEvent(type="message_stop")
        return StreamEvent(type="unknown")


def _serialize_content_block(cb) -> dict[str, Any]:
    if cb.type == "text":
        return {"type": "text", "text": cb.text or ""}
    if cb.type == "thinking":
        return {"type": "thinking", "thinking": cb.thinking or ""}
    if cb.type == "tool_use":
        return {
            "type": "tool_use",
            "id": cb.id or "",
            "name": cb.name or "",
            "input": cb.input or {},
        }
    if cb.type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": cb.tool_use_id or "",
            "content": cb.content or "",
            **({"is_error": True} if cb.is_error else {}),
        }
    return {"type": cb.type}


def _http_error_to_llm(err: httpx.HTTPStatusError) -> LlmError:
    """Convert an httpx HTTP error to an LlmError."""
    status = err.response.status_code
    body = err.response.text
    return LlmError.from_http_response(status, body)
