"""UI state machine — port of `deepseek-tui-core` crate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Pane(Enum):
    CHAT = auto()
    DIFF = auto()
    TASKS = auto()
    AGENTS = auto()
    STATUS = auto()
    JOBS = auto()


@dataclass
class UiState:
    """Application state for the TUI."""
    active_pane: Pane = Pane.CHAT
    paused: bool = False
    last_response_delta: Optional[str] = None
    active_tool: Optional[str] = None
    pending_tasks: int = 0
    active_jobs: int = 0
    pending_approvals: int = 0
    status_line: str = "ready"
    conversation_history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    def snapshot(self) -> str:
        return (
            f"pane={self.active_pane.name};paused={self.paused};"
            f"pending_tasks={self.pending_tasks};active_jobs={self.active_jobs};"
            f"pending_approvals={self.pending_approvals};"
            f"active_tool={self.active_tool or ''};status={self.status_line}"
        )
