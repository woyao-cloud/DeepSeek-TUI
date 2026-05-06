"""LLM client abstraction and DeepSeek chat-completions implementation."""

from __future__ import annotations

import abc
import json
import re
from typing import Any, AsyncIterator, Optional

import httpx
from httpx_sse import aconnect_sse

from .models import (
    ContentBlock,
    Delta,
    Message,
    MessageDelta,
    MessageRequest,
    MessageResponse,
    StreamEvent,
    SystemPrompt,
    Usage,
)
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
_THINKING_PLACEHOLDER = "(reasoning omitted)"
_BARE_HEX_ESCAPE_RE = re.compile(r"x([0-9A-Fa-f]{6})-?")


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

    async def create_message_stream(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        content_index = 0
        text_started = False
        thinking_started = False
        tool_indices: dict[int, int] = {}
        is_reasoning_model = _requires_reasoning_content(request.model)

        yield StreamEvent(
            type="message_start",
            message=MessageResponse(
                id="",
                model=request.model or self._model,
                content=[],
                usage=Usage(),
            ),
        )

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
                        chunk = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue

                    for parsed in self._parse_sse_chunk(
                        chunk=chunk,
                        content_index_ref=[content_index],
                        text_started_ref=[text_started],
                        thinking_started_ref=[thinking_started],
                        tool_indices=tool_indices,
                        is_reasoning_model=is_reasoning_model,
                    ):
                        content_index = parsed["content_index"]
                        text_started = parsed["text_started"]
                        thinking_started = parsed["thinking_started"]
                        yield parsed["event"]
        except httpx.HTTPStatusError as e:
            raise _http_error_to_llm(e) from e
        except httpx.RequestError as e:
            raise LlmError.network(f"Stream failed: {e}") from e

        if thinking_started:
            yield StreamEvent(type="content_block_stop", index=content_index)
        if text_started:
            yield StreamEvent(type="content_block_stop", index=content_index)
        for tool_block_index in sorted(tool_indices.values()):
            yield StreamEvent(type="content_block_stop", index=tool_block_index)
        yield StreamEvent(type="message_stop")

    def _build_payload(self, request: MessageRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": self._serialize_messages(request),
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.thinking is not None and request.reasoning_effort is None:
            payload["thinking"] = request.thinking
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
            payload["thinking"] = {"type": "disabled"} if _thinking_disabled(
                request.reasoning_effort
            ) else {"type": "enabled"}
        if request.tools is not None:
            payload["tools"] = self._serialize_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = _map_tool_choice(request.tool_choice)
        return payload

    @staticmethod
    def _serialize_messages(request: MessageRequest) -> list[dict[str, Any]]:
        messages = request.messages
        include_reasoning = _should_replay_reasoning_content(
            request.model,
            request.reasoning_effort,
        )
        serialized: list[dict[str, Any]] = []
        pending_tool_calls: set[str] = set()

        instructions = _system_to_instructions(request.system)
        if instructions:
            serialized.append({"role": "system", "content": instructions})

        for message in messages:
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_call_ids: list[str] = []
            tool_results: list[tuple[str, dict[str, Any]]] = []

            for block in message.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "thinking" and block.thinking:
                    thinking_parts.append(block.thinking)
                elif block.type == "tool_use":
                    tool_id = block.id or "tool_call"
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": _to_api_tool_name(block.name or "tool"),
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                    )
                    tool_call_ids.append(tool_id)
                elif block.type == "tool_result" and block.tool_use_id:
                    tool_results.append(
                        (
                            block.tool_use_id,
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": block.content or "",
                            },
                        )
                    )

            if message.role == "assistant":
                content = "\n".join(text_parts)
                reasoning = "\n".join(thinking_parts)
                has_text = bool(content.strip())
                has_tool_calls = bool(tool_calls)
                has_reasoning = include_reasoning and bool(reasoning.strip())
                if include_reasoning and not has_reasoning:
                    reasoning = _THINKING_PLACEHOLDER
                    has_reasoning = True
                if not has_text and not has_tool_calls and not has_reasoning:
                    pending_tool_calls.clear()
                    continue

                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": content if has_text else ("" if has_reasoning else None),
                }
                if has_reasoning:
                    assistant["reasoning_content"] = reasoning
                if has_tool_calls:
                    assistant["tool_calls"] = tool_calls
                    pending_tool_calls = set(tool_call_ids)
                else:
                    pending_tool_calls.clear()
                serialized.append(assistant)

            elif message.role == "user":
                content = "\n".join(text_parts)
                if content.strip():
                    serialized.append({"role": "user", "content": content})

            if tool_results:
                if not pending_tool_calls:
                    continue
                for tool_call_id, tool_message in tool_results:
                    if tool_call_id in pending_tool_calls:
                        pending_tool_calls.remove(tool_call_id)
                        serialized.append(tool_message)
            elif message.role != "assistant":
                pending_tool_calls.clear()

        return _strip_orphaned_tool_rounds(serialized)

    @staticmethod
    def _serialize_tools(tools: list) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name", "tool")
                description = tool.get("description", "")
                schema = tool.get("input_schema", {})
            else:
                name = tool.name
                description = tool.description
                schema = tool.input_schema
            serialized.append(
                {
                    "type": "function",
                    "function": {
                        "name": _to_api_tool_name(name),
                        "description": description,
                        "parameters": schema,
                    },
                }
            )
        return serialized

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> MessageResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content_blocks: list[ContentBlock] = []

        reasoning = _reasoning_field(message)
        if reasoning and reasoning.strip():
            content_blocks.append(ContentBlock(type="thinking", thinking=reasoning))

        content_raw = message.get("content")
        if isinstance(content_raw, str) and content_raw.strip():
            content_blocks.append(ContentBlock(type="text", text=content_raw))

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            raw_arguments = function.get("arguments", "{}")
            try:
                parsed_arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                parsed_arguments = raw_arguments
            content_blocks.append(
                ContentBlock(
                    type="tool_use",
                    id=tool_call.get("id", ""),
                    name=_from_api_tool_name(function.get("name", "tool")),
                    input=parsed_arguments,
                )
            )

        return MessageResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content_blocks,
            stop_reason=choice.get("finish_reason"),
            usage=_parse_usage(data.get("usage")),
        )

    @staticmethod
    def _parse_sse_chunk(
        chunk: dict[str, Any],
        content_index_ref: list[int],
        text_started_ref: list[bool],
        thinking_started_ref: list[bool],
        tool_indices: dict[int, int],
        is_reasoning_model: bool,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        content_index = content_index_ref[0]
        text_started = text_started_ref[0]
        thinking_started = thinking_started_ref[0]

        choices = chunk.get("choices")
        if not choices:
            usage = chunk.get("usage")
            if usage is not None:
                events.append(
                    {
                        "content_index": content_index,
                        "text_started": text_started,
                        "thinking_started": thinking_started,
                        "event": StreamEvent(
                            type="message_delta",
                            message_delta=MessageDelta(),
                            usage=_parse_usage(usage),
                        ),
                    }
                )
            return events

        for choice in choices:
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            reasoning = _reasoning_field(delta)
            if is_reasoning_model and reasoning:
                if not thinking_started:
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": text_started,
                            "thinking_started": True,
                            "event": StreamEvent(
                                type="content_block_start",
                                index=content_index,
                                content_block=ContentBlock(
                                    type="thinking",
                                    thinking="",
                                ),
                            ),
                        }
                    )
                    thinking_started = True
                events.append(
                    {
                        "content_index": content_index,
                        "text_started": text_started,
                        "thinking_started": thinking_started,
                        "event": StreamEvent(
                            type="content_block_delta",
                            index=content_index,
                            delta=Delta(type="thinking_delta", thinking=reasoning),
                        ),
                    }
                )

            content = delta.get("content")
            if isinstance(content, str) and content:
                if thinking_started:
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": text_started,
                            "thinking_started": False,
                            "event": StreamEvent(
                                type="content_block_stop",
                                index=content_index,
                            ),
                        }
                    )
                    content_index += 1
                    thinking_started = False
                if not text_started:
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": True,
                            "thinking_started": thinking_started,
                            "event": StreamEvent(
                                type="content_block_start",
                                index=content_index,
                                content_block=ContentBlock(type="text", text=""),
                            ),
                        }
                    )
                    text_started = True
                events.append(
                    {
                        "content_index": content_index,
                        "text_started": text_started,
                        "thinking_started": thinking_started,
                        "event": StreamEvent(
                            type="content_block_delta",
                            index=content_index,
                            delta=Delta(type="text_delta", text=content),
                        ),
                    }
                )

            for tool_call in delta.get("tool_calls") or []:
                tool_call_index = int(tool_call.get("index", 0))
                tool_block_index = tool_indices.get(tool_call_index)
                if tool_block_index is None:
                    if text_started:
                        events.append(
                            {
                                "content_index": content_index,
                                "text_started": False,
                                "thinking_started": thinking_started,
                                "event": StreamEvent(
                                    type="content_block_stop",
                                    index=content_index,
                                ),
                            }
                        )
                        content_index += 1
                        text_started = False
                    if thinking_started:
                        events.append(
                            {
                                "content_index": content_index,
                                "text_started": text_started,
                                "thinking_started": False,
                                "event": StreamEvent(
                                    type="content_block_stop",
                                    index=content_index,
                                ),
                            }
                        )
                        content_index += 1
                        thinking_started = False

                    tool_block_index = content_index
                    function = tool_call.get("function") or {}
                    tool_id = tool_call.get("id") or f"call_{tool_block_index}"
                    events.append(
                        {
                            "content_index": tool_block_index + 1,
                            "text_started": text_started,
                            "thinking_started": thinking_started,
                            "event": StreamEvent(
                                type="content_block_start",
                                index=tool_block_index,
                                content_block=ContentBlock(
                                    type="tool_use",
                                    id=tool_id,
                                    name=_from_api_tool_name(function.get("name", "tool")),
                                    input={},
                                ),
                            ),
                        }
                    )
                    tool_indices[tool_call_index] = tool_block_index
                    content_index += 1

                arguments = (
                    (tool_call.get("function") or {}).get("arguments")
                    if isinstance(tool_call, dict)
                    else None
                )
                if isinstance(arguments, str) and arguments:
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": text_started,
                            "thinking_started": thinking_started,
                            "event": StreamEvent(
                                type="content_block_delta",
                                index=tool_block_index,
                                delta=Delta(
                                    type="input_json_delta",
                                    partial_json=arguments,
                                ),
                            ),
                        }
                    )

            if finish_reason is not None:
                if text_started:
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": False,
                            "thinking_started": thinking_started,
                            "event": StreamEvent(
                                type="content_block_stop",
                                index=content_index,
                            ),
                        }
                    )
                    text_started = False
                if thinking_started:
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": text_started,
                            "thinking_started": False,
                            "event": StreamEvent(
                                type="content_block_stop",
                                index=content_index,
                            ),
                        }
                    )
                    thinking_started = False

                for tool_block_index in sorted(tool_indices.values()):
                    events.append(
                        {
                            "content_index": content_index,
                            "text_started": text_started,
                            "thinking_started": thinking_started,
                            "event": StreamEvent(
                                type="content_block_stop",
                                index=tool_block_index,
                            ),
                        }
                    )
                tool_indices.clear()
                events.append(
                    {
                        "content_index": content_index,
                        "text_started": text_started,
                        "thinking_started": thinking_started,
                        "event": StreamEvent(
                            type="message_delta",
                            message_delta=MessageDelta(stop_reason=finish_reason),
                            stop_reason=finish_reason,
                            usage=_parse_usage(chunk.get("usage"))
                            if chunk.get("usage") is not None
                            else None,
                        ),
                    }
                )

        content_index_ref[0] = content_index
        text_started_ref[0] = text_started
        thinking_started_ref[0] = thinking_started
        return events


def _system_to_instructions(system: Optional[SystemPrompt]) -> Optional[str]:
    if system is None:
        return None
    if system.text and system.text.strip():
        return system.text
    if system.blocks:
        joined = "\n\n---\n\n".join(block.text for block in system.blocks if block.text)
        return joined or None
    return None


def _thinking_disabled(reasoning_effort: str) -> bool:
    normalized = reasoning_effort.strip().lower()
    return normalized in {"off", "disabled", "none", "false"}


def _requires_reasoning_content(model: str) -> bool:
    lower = model.lower()
    return (
        "deepseek-v3.2" in lower
        or "deepseek-v4" in lower
        or "reasoner" in lower
        or "-reasoning" in lower
        or "-thinking" in lower
        or bool(re.search(r"deepseek-r\d", lower))
    )


def _should_replay_reasoning_content(
    model: str,
    reasoning_effort: Optional[str],
) -> bool:
    if reasoning_effort is not None and _thinking_disabled(reasoning_effort):
        return False
    return _requires_reasoning_content(model)


def _map_tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        if choice == "any":
            return "auto"
        return choice
    if not isinstance(choice, dict):
        return choice

    choice_type = choice.get("type")
    if choice_type in {"auto", "none"}:
        return choice_type
    if choice_type == "any":
        return "auto"
    if choice_type == "tool":
        name = choice.get("name")
        if isinstance(name, str) and name:
            return {
                "type": "function",
                "function": {"name": _to_api_tool_name(name)},
            }
    return choice


def _parse_usage(usage: Optional[dict[str, Any]]) -> Usage:
    if not isinstance(usage, dict):
        return Usage()
    completion_details = usage.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        output_tokens=int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        ),
        prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
        prompt_cache_miss_tokens=usage.get("prompt_cache_miss_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
    )


def _reasoning_field(value: dict[str, Any]) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    reasoning = value.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning
    reasoning = value.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    return None


def _strip_orphaned_tool_rounds(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(messages)
    i = 0
    while i < len(out):
        is_assistant_with_tools = (
            out[i].get("role") == "assistant"
            and out[i].get("tool_calls") is not None
        )
        if not is_assistant_with_tools:
            i += 1
            continue

        tool_calls = out[i].get("tool_calls") or []
        expected_ids = {
            call.get("id")
            for call in tool_calls
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        }

        found_ids: set[str] = set()
        tool_result_end = i + 1
        while tool_result_end < len(out):
            if out[tool_result_end].get("role") == "tool":
                tool_id = out[tool_result_end].get("tool_call_id")
                if isinstance(tool_id, str):
                    found_ids.add(tool_id)
                tool_result_end += 1
            else:
                break

        scan = tool_result_end
        while scan < len(out):
            if out[scan].get("role") == "assistant":
                break
            if out[scan].get("role") == "tool":
                tool_id = out[scan].get("tool_call_id")
                if isinstance(tool_id, str):
                    found_ids.add(tool_id)
            scan += 1

        if expected_ids.issubset(found_ids):
            i += 1
            continue

        if isinstance(out[i], dict):
            out[i].pop("tool_calls", None)

        assistant_content = out[i].get("content")
        assistant_content_empty = assistant_content is None or (
            isinstance(assistant_content, str) and not assistant_content
        )
        if assistant_content_empty:
            for j in range(len(out) - 1, i, -1):
                if out[j].get("role") == "tool":
                    tool_id = out[j].get("tool_call_id")
                    if isinstance(tool_id, str) and tool_id in expected_ids:
                        out.pop(j)
            out.pop(i)
            i = max(i - 1, 0)
            continue

        if tool_result_end > i + 1:
            del out[i + 1:tool_result_end]

        for j in range(len(out) - 1, i, -1):
            if out[j].get("role") == "tool":
                tool_id = out[j].get("tool_call_id")
                if isinstance(tool_id, str) and tool_id in expected_ids:
                    out.pop(j)

        i += 1

    return out


def _to_api_tool_name(name: str) -> str:
    out: list[str] = []
    for char in name:
        if char.isascii() and (char.isalnum() or char == "_"):
            out.append(char)
        elif char == "-":
            out.append("--")
        else:
            out.append(f"-x{ord(char):06X}-")
    return "".join(out)


def _from_api_tool_name(name: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(name):
        char = name[index]
        if char != "-":
            out.append(char)
            index += 1
            continue
        if index + 1 < len(name) and name[index + 1] == "-":
            out.append("-")
            index += 2
            continue
        if index + 8 < len(name) and name[index + 1] == "x":
            hex_value = name[index + 2:index + 8]
            try:
                decoded = chr(int(hex_value, 16))
            except ValueError:
                out.append("-")
                index += 1
                continue
            out.append(decoded)
            index += 8
            if index < len(name) and name[index] == "-":
                index += 1
            continue
        out.append("-")
        index += 1

    def decode_bare_hex(match: re.Match[str]) -> str:
        try:
            decoded = chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)
        if decoded.isascii() and (decoded.isalnum() or decoded in {"_", "-"}):
            return match.group(0)
        return decoded

    return _BARE_HEX_ESCAPE_RE.sub(decode_bare_hex, "".join(out))


def _http_error_to_llm(err: httpx.HTTPStatusError) -> LlmError:
    """Convert an httpx HTTP error to an LlmError."""
    status = err.response.status_code
    body = err.response.text
    return LlmError.from_http_response(status, body)
