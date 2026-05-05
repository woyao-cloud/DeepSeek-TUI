"""Core protocol types — port of `deepseek-protocol` crate.

All types use Pydantic v2 models for serialization/deserialization.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────

class ThreadStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ARCHIVED = "archived"


class SessionSource(str, Enum):
    INTERACTIVE = "interactive"
    RESUME = "resume"
    FORK = "fork"
    API = "api"
    UNKNOWN = "unknown"


class ToolKind(str, Enum):
    FUNCTION = "function"
    MCP = "mcp"


class NetworkPolicyRuleAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class McpStartupStatus(str, Enum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Core Types ────────────────────────────────────────────────────────

class Thread(BaseModel):
    """A conversation thread."""
    id: str
    preview: str
    ephemeral: bool = False
    model_provider: str
    created_at: int
    updated_at: int
    status: ThreadStatus = ThreadStatus.RUNNING
    path: Optional[Path] = None
    cwd: Path
    cli_version: str
    source: SessionSource = SessionSource.UNKNOWN
    name: Optional[str] = None


# ── Thread Params ─────────────────────────────────────────────────────

class ThreadStartParams(BaseModel):
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[Path] = None
    persist_extended_history: bool = False


class ThreadResumeParams(BaseModel):
    thread_id: str
    history: Optional[list[Any]] = None
    path: Optional[Path] = None
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[Path] = None
    approval_policy: Optional[str] = None
    sandbox: Optional[str] = None
    config: Optional[Any] = None
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None
    personality: Optional[str] = None
    persist_extended_history: bool = False


class ThreadForkParams(BaseModel):
    thread_id: str
    path: Optional[Path] = None
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[Path] = None
    approval_policy: Optional[str] = None
    sandbox: Optional[str] = None
    config: Optional[Any] = None
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None
    persist_extended_history: bool = False


class ThreadListParams(BaseModel):
    include_archived: bool = False
    limit: Optional[int] = None


class ThreadReadParams(BaseModel):
    thread_id: str


class ThreadSetNameParams(BaseModel):
    thread_id: str
    name: str


# ── Thread Request / Response ────────────────────────────────────────

class ThreadRequest(BaseModel):
    """Union-like request via discriminator field `kind`."""
    kind: str = Field(alias="kind")
    # Create
    metadata: Any = None
    # Start / Resume / Fork
    thread_id: Optional[str] = None
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[Path] = None
    persist_extended_history: bool = False
    # Resume-only
    history: Optional[list[Any]] = None
    path: Optional[Path] = None
    approval_policy: Optional[str] = None
    sandbox: Optional[str] = None
    config: Optional[Any] = None
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None
    personality: Optional[str] = None
    # List
    include_archived: bool = False
    limit: Optional[int] = None
    # SetName
    name: Optional[str] = None
    # Message
    input: Optional[str] = None

    model_config = {"populate_by_name": True}


class ThreadResponse(BaseModel):
    thread_id: str
    status: str
    thread: Optional[Thread] = None
    threads: list[Thread] = Field(default_factory=list)
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[Path] = None
    approval_policy: Optional[str] = None
    sandbox: Optional[str] = None
    events: list[EventFrame] = Field(default_factory=list)
    data: Any = Field(default_factory=dict)


# ── App / Prompt ─────────────────────────────────────────────────────

class AppRequest(BaseModel):
    kind: str
    key: Optional[str] = None
    value: Optional[str] = None


class AppResponse(BaseModel):
    ok: bool
    data: Any = Field(default_factory=dict)
    events: list[EventFrame] = Field(default_factory=list)


class PromptRequest(BaseModel):
    thread_id: Optional[str] = None
    prompt: str
    model: Optional[str] = None


class PromptResponse(BaseModel):
    output: str
    model: str
    events: list[EventFrame] = Field(default_factory=list)


# ── Tool Types ────────────────────────────────────────────────────────

class AskForApproval(str, Enum):
    UNLESS_TRUSTED = "unless_trusted"
    ON_FAILURE = "on_failure"
    ON_REQUEST = "on_request"
    NEVER = "never"

    @classmethod
    def reject(cls) -> AskForApproval:
        return cls.ON_REQUEST  # simplified


class LocalShellParams(BaseModel):
    command: str
    cwd: Optional[str] = None
    timeout_ms: Optional[int] = None


class ToolPayload(BaseModel):
    """Union type via `type` discriminator."""
    type: str
    # Function
    arguments: Optional[str] = None
    # Custom
    input: Optional[str] = None
    # LocalShell
    params: Optional[LocalShellParams | dict] = None
    # MCP
    server: Optional[str] = None
    tool: Optional[str] = None
    raw_arguments: Optional[Any] = None
    raw_tool_call_id: Optional[str] = None


class ToolOutput(BaseModel):
    type: str
    body: Optional[Any] = None
    success: bool = True
    result: Optional[Any] = None  # for MCP


# ── Approval / Policy ─────────────────────────────────────────────────

class NetworkPolicyAmendment(BaseModel):
    host: str
    action: NetworkPolicyRuleAction


class ReviewDecision(BaseModel):
    type: str
    host: Optional[str] = None
    action: Optional[NetworkPolicyRuleAction] = None


class McpStartupUpdateEvent(BaseModel):
    server_name: str
    status: McpStartupStatus


class McpStartupFailure(BaseModel):
    server_name: str
    error: str


class McpStartupCompleteEvent(BaseModel):
    ready: list[str] = Field(default_factory=list)
    failed: list[McpStartupFailure] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)


class NetworkApprovalContext(BaseModel):
    host: str
    protocol: str


class ExecApprovalRequestEvent(BaseModel):
    call_id: str
    approval_id: str
    turn_id: str
    command: str
    cwd: str
    reason: str
    network_approval_context: Optional[NetworkApprovalContext] = None
    proposed_execpolicy_amendment: list[str] = Field(default_factory=list)
    proposed_network_policy_amendments: list[NetworkPolicyAmendment] = Field(default_factory=list)
    additional_permissions: list[str] = Field(default_factory=list)
    available_decisions: list[ReviewDecision] = Field(default_factory=list)


# ── Event Frame ───────────────────────────────────────────────────────

class EventFrame(BaseModel):
    """Union of all event frame types, discriminated by `event`."""
    event: str
    response_id: Optional[str] = None
    delta: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[Any] = None
    output: Optional[Any] = None
    message: Optional[str] = None
    update: Optional[McpStartupUpdateEvent] = None
    summary: Optional[McpStartupCompleteEvent] = None
    request: Optional[ExecApprovalRequestEvent] = None
    command: Optional[str] = None
    cwd: Optional[str] = None
    exit_code: Optional[int] = None
    path: Optional[str] = None
    ok: Optional[bool] = None
    turn_id: Optional[str] = None
    reason: Optional[str] = None
    server_name: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────

def truncate_preview(value: str, max_chars: int = 120) -> str:
    """Truncate a string for use as a thread preview."""
    if len(value) <= max_chars:
        return value
    return value[:max_chars]
