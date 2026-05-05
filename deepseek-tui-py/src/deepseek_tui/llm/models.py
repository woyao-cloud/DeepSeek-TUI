"""LLM API request/response models — port of `crates/tui/src/models.rs`.

MessageRequest, ContentBlock, StreamEvent, and related API types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Constants ─────────────────────────────────────────────────────────

LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS = 128_000
DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_COMPACTION_TOKEN_THRESHOLD = 102_400  # 80% of 128K
COMPACTION_THRESHOLD_PERCENT = 80


# ── Core Types ────────────────────────────────────────────────────────

class CacheControl(BaseModel):
    type: str = "ephemeral"


class SystemPrompt(BaseModel):
    """System prompt — text string or structured blocks."""
    text: Optional[str] = None
    blocks: Optional[list[SystemBlock]] = None


class SystemBlock(BaseModel):
    type: str
    text: str
    cache_control: Optional[CacheControl] = None


class ContentBlock(BaseModel):
    """A single content block inside a message (text, thinking, tool_use, tool_result)."""
    type: str = "text"
    text: Optional[str] = None
    thinking: Optional[str] = None
    id: Optional[str] = None  # tool_use id
    name: Optional[str] = None  # tool_use name
    input: Optional[Any] = None  # tool_use input
    tool_use_id: Optional[str] = None  # tool_result
    content: Optional[str] = None  # tool_result content
    is_error: Optional[bool] = None
    cache_control: Optional[CacheControl] = None
    caller: Optional[dict[str, str]] = None


class ContentBlockData(BaseModel):
    """Simpler data variant for storing block content."""
    type: str
    text: Optional[str] = None
    thinking: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None


class Message(BaseModel):
    """A chat message with role and content blocks."""
    role: str  # "user" | "assistant" | "system"
    content: list[ContentBlock] = Field(default_factory=list)


class Tool(BaseModel):
    """Tool definition exposed to the model."""
    type: Optional[str] = None
    name: str
    description: str = ""
    input_schema: Any = Field(default_factory=dict)
    allowed_callers: Optional[list[str]] = None
    defer_loading: Optional[bool] = None
    input_examples: Optional[list[Any]] = None
    strict: Optional[bool] = None
    cache_control: Optional[CacheControl] = None


class Usage(BaseModel):
    """Token usage metadata."""
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_cache_hit_tokens: Optional[int] = None
    prompt_cache_miss_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    reasoning_replay_tokens: Optional[int] = None


# ── Request / Response ────────────────────────────────────────────────

class MessageRequest(BaseModel):
    """Request payload for sending a message to the API."""
    model: str
    messages: list[Message]
    max_tokens: int = 8192
    system: Optional[SystemPrompt] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[Any] = None
    thinking: Optional[Any] = None
    reasoning_effort: Optional[str] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class MessageResponse(BaseModel):
    """Response payload from the API."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)


# ── Streaming Events ──────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """Streaming event types for SSE responses."""
    type: str  # "message_start", "content_block_start", etc.
    message: Optional[MessageResponse] = None
    index: Optional[int] = None
    content_block: Optional[ContentBlock] = None
    delta: Optional[Delta] = None
    usage: Optional[Usage] = None


class Delta(BaseModel):
    """Delta event payloads during streaming."""
    type: str  # "text_delta", "thinking_delta", "input_json_delta"
    text: Optional[str] = None
    thinking: Optional[str] = None
    partial_json: Optional[str] = None


# ── Context Window Helpers ───────────────────────────────────────────

def context_window_for_model(model: str) -> Optional[int]:
    """Map known models to their approximate context window sizes."""
    lower = model.lower()
    if "deepseek" in lower:
        if lower in ("deepseek-chat", "deepseek-reasoner", "deepseek-r1",
                      "deepseek-v3", "deepseek-v3.2"):
            return DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS
        if "v4" in lower:
            return DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS
        return LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS
    if "claude" in lower:
        return 200_000
    return None


def compaction_threshold_for_model(model: str) -> int:
    """Derive a compaction token threshold from model context window."""
    window = context_window_for_model(model)
    if window is None:
        return DEFAULT_COMPACTION_TOKEN_THRESHOLD
    return int(window * COMPACTION_THRESHOLD_PERCENT / 100)
