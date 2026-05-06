"""LSP integration — Diagnostic types and rendering.

Port of `crates/tui/src/lsp/diagnostics.rs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional


class Severity(IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class Diagnostic:
    """A single diagnostic item (error, warning, etc.)."""
    line: int
    column: int
    severity: Severity
    message: str
    code: Optional[str] = None
    source: Optional[str] = None


@dataclass
class DiagnosticBlock:
    """Diagnostics for a single file."""
    file: Path
    items: list[Diagnostic] = field(default_factory=list)

    def truncate(self, max_items: int) -> None:
        if len(self.items) > max_items:
            self.items = self.items[:max_items]

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def render(self) -> str:
        """Render diagnostics as a formatted block."""
        if not self.items:
            return ""
        lines = [f"file=\"{self.file}\""]
        severity_labels = {
            Severity.ERROR: "ERROR",
            Severity.WARNING: "WARN",
            Severity.INFORMATION: "INFO",
            Severity.HINT: "HINT",
        }
        for d in self.items:
            label = severity_labels.get(d.severity, "?")
            code = f" ({d.code})" if d.code else ""
            lines.append(f"  {label} [{d.line}:{d.column}]{code} {d.message}")
        return "\n".join(lines)


def render_blocks(blocks: list[DiagnosticBlock]) -> str:
    """Render multiple diagnostic blocks into a single string."""
    parts = [b.render() for b in blocks if not b.is_empty()]
    return "\n---\n".join(parts)
