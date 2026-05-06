"""Revert turn tool — model-visible tool for workspace rollback.

Port of `crates/tui/src/tools/revert_turn.rs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ToolSpec, ToolResult, ToolCapability, ApprovalRequirement
from .context import ToolContext
from .snapshot import auto_restore, auto_snapshot


class RevertTurnTool(ToolSpec):
    name = "revert_turn"
    description = "Revert the workspace to a previous snapshot state."
    input_schema = {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "Number of snapshots to revert (default: 1, meaning last snapshot)",
            },
        },
        "required": [],
    }

    def capabilities(self):
        return [ToolCapability.WRITES_FILES]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        n = input.get("n", 1)
        workspace = context.params.workspace

        try:
            from .snapshot import SnapshotRepo
            repo = SnapshotRepo(workspace)
            snapshots = repo.list(limit=n)

            if not snapshots:
                return ToolResult.error("No snapshots available to revert to.")

            target = snapshots[0] if n <= 1 else snapshots[min(n - 1, len(snapshots) - 1)]
            repo.restore(target.id)

            return ToolResult.success(
                f"Reverted to snapshot: {target.label} ({target.id.sha[:12]})"
            )
        except Exception as e:
            return ToolResult.error(f"Revert failed: {e}")


async def pre_turn_snapshot(workspace: Path, turn_seq: int) -> None:
    """Take a pre-turn snapshot. Non-fatal on failure."""
    try:
        auto_snapshot(workspace, f"pre-turn:{turn_seq}")
    except Exception:
        pass  # Snapshot failures are non-fatal


async def post_turn_snapshot(workspace: Path, turn_seq: int) -> None:
    """Take a post-turn snapshot. Non-fatal on failure."""
    try:
        auto_snapshot(workspace, f"post-turn:{turn_seq}")
    except Exception:
        pass
