"""ToolContext — execution context passed to every tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolParams:
    """Parameters available to tools during execution."""
    workspace: Path = Path(".")
    cwd: Path = Path(".")
    shell_allowed: bool = False
    network_allowed: bool = False
    sandbox_mode: Optional[str] = None
    approval_policy: Optional[str] = None


@dataclass
class ToolContext:
    """Rich execution context passed to every tool execution."""
    params: ToolParams = field(default_factory=ToolParams)
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, workspace: Path) -> ToolContext:
        return cls(params=ToolParams(workspace=workspace, cwd=workspace))
