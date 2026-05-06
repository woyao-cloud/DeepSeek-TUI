"""Remember tool — model-callable bullet-add into the user memory file.

Port of `crates/tui/src/tools/remember.rs`.
"""

from __future__ import annotations

from typing import Any

from .base import ToolSpec, ToolResult, ToolError, ToolCapability, ApprovalRequirement
from .context import ToolContext
from .memory import append_entry


class RememberTool(ToolSpec):
    name = "remember"
    description = (
        "Append a durable note to the user memory file so it surfaces in "
        "future sessions. Use this when the user states a preference, a "
        "convention they want enforced, or a fact about themselves or "
        "their workflow that you should not have to relearn next time. "
        "Keep notes terse (one sentence). Don't store secrets, transient "
        "tasks, or reasoning scratch."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "The single-sentence durable note to remember.",
            },
        },
        "required": ["note"],
    }

    def capabilities(self):
        return [ToolCapability.WRITES_FILES]

    def approval_requirement(self):
        return ApprovalRequirement.AUTO

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        note = input.get("note", "")
        if not note:
            return ToolResult.error("Missing required field 'note'")

        # Try to get memory path from context
        memory_path = context.extra.get("memory_path")
        if memory_path is None:
            return ToolResult.error(
                "User memory is disabled — enable it in config to use this tool."
            )

        from pathlib import Path
        try:
            append_entry(Path(memory_path), note)
            return ToolResult.success(f"remembered: {note.lstrip('#').strip()}")
        except Exception as e:
            return ToolResult.error(f"Failed to append to memory: {e}")
