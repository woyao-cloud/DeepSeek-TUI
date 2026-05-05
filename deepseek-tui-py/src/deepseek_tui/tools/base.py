"""Base tool types — port of `deepseek-tools` ToolSpec, ToolResult, ToolError, helpers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import json as _json


# ── Enums ─────────────────────────────────────────────────────────────

class ToolCapability(Enum):
    READ_ONLY = auto()
    WRITES_FILES = auto()
    EXECUTES_CODE = auto()
    NETWORK = auto()
    SANDBOXABLE = auto()
    REQUIRES_APPROVAL = auto()


class ApprovalRequirement(Enum):
    AUTO = "auto"
    SUGGEST = "suggest"
    REQUIRED = "required"


class ToolCallSource(Enum):
    DIRECT = "direct"
    JS_REPL = "js_repl"


# ── ToolError ─────────────────────────────────────────────────────────

class ToolError(Exception):
    """Errors that can occur during tool execution."""

    def __init__(self, kind: str, message: str = "") -> None:
        self.kind = kind
        self.message = message
        super().__init__(self._format())

    def _format(self) -> str:
        if self.kind == "invalid_input":
            return f"Failed to validate input: {self.message}"
        if self.kind == "missing_field":
            return f"Failed to validate input: missing required field '{self.message}'"
        if self.kind == "path_escape":
            return f"Failed to resolve path '{self.message}': path escapes workspace"
        if self.kind == "execution_failed":
            return f"Failed to execute tool: {self.message}"
        if self.kind == "timeout":
            return f"Failed to execute tool: operation timed out after {self.message}s"
        if self.kind == "not_available":
            return f"Failed to locate tool: {self.message}"
        if self.kind == "permission_denied":
            return f"Failed to authorize tool execution: {self.message}"
        return f"Tool error ({self.kind}): {self.message}"

    @classmethod
    def invalid_input(cls, message: str) -> ToolError:
        return cls("invalid_input", message)

    @classmethod
    def missing_field(cls, field: str) -> ToolError:
        return cls("missing_field", field)

    @classmethod
    def path_escape(cls, path: str) -> ToolError:
        return cls("path_escape", path)

    @classmethod
    def execution_failed(cls, message: str) -> ToolError:
        return cls("execution_failed", message)

    @classmethod
    def timeout(cls, seconds: int) -> ToolError:
        return cls("timeout", str(seconds))

    @classmethod
    def not_available(cls, message: str) -> ToolError:
        return cls("not_available", message)

    @classmethod
    def permission_denied(cls, message: str) -> ToolError:
        return cls("permission_denied", message)


# ── ToolResult ────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    content: str
    success: bool = True
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def success(cls, content: str) -> ToolResult:
        return cls(content=content, success=True)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=message, success=False)

    @classmethod
    def json(cls, value: Any) -> ToolResult:
        return cls(content=_json.dumps(value, indent=2), success=True)

    def with_metadata(self, metadata: dict[str, Any]) -> ToolResult:
        self.metadata = metadata
        return self


# ── ToolSpec ABC ──────────────────────────────────────────────────────

class ToolSpec(abc.ABC):
    """Abstract base for all tool implementations."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def input_schema(self) -> dict[str, Any]:
        ...

    def capabilities(self) -> list[ToolCapability]:
        return [ToolCapability.READ_ONLY]

    def approval_requirement(self) -> ApprovalRequirement:
        return ApprovalRequirement.AUTO

    def is_read_only(self) -> bool:
        return ToolCapability.READ_ONLY in self.capabilities()

    def is_mutating(self) -> bool:
        return ToolCapability.WRITES_FILES in self.capabilities() or ToolCapability.EXECUTES_CODE in self.capabilities()

    def defer_loading(self) -> bool:
        return False

    @abc.abstractmethod
    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


# ── Input Validation Helpers ─────────────────────────────────────────

def required_str(input: dict[str, Any], field: str) -> str:
    value = input.get(field)
    if value is None or not isinstance(value, str):
        provided = list(input.keys())
        if not provided:
            raise ToolError.missing_field(field)
        raise ToolError.invalid_input(
            f"missing required field '{field}'. Input provided: {', '.join(provided)}"
        )
    return value


def optional_str(input: dict[str, Any], field: str) -> Optional[str]:
    value = input.get(field)
    if isinstance(value, str):
        return value
    return None


def required_u64(input: dict[str, Any], field: str) -> int:
    value = input.get(field)
    if value is None or not isinstance(value, (int, float)):
        raise ToolError.missing_field(field)
    return int(value)


def optional_u64(input: dict[str, Any], field: str, default: int = 0) -> int:
    value = input.get(field)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def optional_bool(input: dict[str, Any], field: str, default: bool = False) -> bool:
    value = input.get(field)
    if isinstance(value, bool):
        return value
    return default
