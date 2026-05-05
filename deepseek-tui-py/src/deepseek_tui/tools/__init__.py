"""Tool system — port of `deepseek-tools` crate and `crates/tui/src/tools/`.

Defines ToolSpec ABC, ToolResult, ToolError, ToolRegistry, and all built-in tools.
"""

from .base import (
    ToolSpec,
    ToolResult,
    ToolError,
    ToolCapability,
    ApprovalRequirement,
    ToolCallSource,
    required_str,
    optional_str,
    required_u64,
    optional_u64,
    optional_bool,
)
from .context import ToolContext, ToolParams
from .registry import ToolRegistry, ToolRegistryBuilder

__all__ = [
    "ToolSpec",
    "ToolResult",
    "ToolError",
    "ToolCapability",
    "ApprovalRequirement",
    "ToolCallSource",
    "ToolContext",
    "ToolParams",
    "ToolRegistry",
    "ToolRegistryBuilder",
    "required_str",
    "optional_str",
    "required_u64",
    "optional_u64",
    "optional_bool",
]
