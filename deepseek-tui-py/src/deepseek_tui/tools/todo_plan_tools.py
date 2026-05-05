"""Todo/plan tools — checklist_write/add/update/list, update_plan.

Port of `crates/tui/src/tools/todo.rs` and `plan.rs`.
"""

from __future__ import annotations

from typing import Any

from .base import ToolSpec, ToolResult, ToolCapability
from .context import ToolContext


# ── In-memory shared state ───────────────────────────────────────────

class _TodoList:
    """Simple in-memory todo list."""
    def __init__(self) -> None:
        self._items: list[dict] = []

    def write(self, items: list[dict]) -> None:
        self._items = list(items)

    def add(self, item: dict) -> None:
        self._items.append(item)

    def update(self, index: int, status: str) -> None:
        if 0 <= index < len(self._items):
            self._items[index]["status"] = status

    def list(self) -> list[dict]:
        return list(self._items)


class _PlanState:
    """Simple in-memory plan."""
    def __init__(self) -> None:
        self._steps: list[dict] = []

    def update(self, steps: list[dict]) -> None:
        self._steps = list(steps)

    def list(self) -> list[dict]:
        return list(self._steps)


# Global singletons
_global_todos = _TodoList()
_global_plan = _PlanState()


class TodoWriteTool(ToolSpec):
    def __init__(self, name: str = "todo_write") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Replace the active todo list with a new set of items."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        items = input.get("todos", [])
        _global_todos.write(items)
        return ToolResult.success(f"Todo list updated with {len(items)} items.")


class TodoAddTool(ToolSpec):
    def __init__(self, name: str = "todo_add") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Add a new item to the todo list."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            },
            "required": ["content", "status"],
        }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        _global_todos.add(input)
        return ToolResult.success("Item added to todo list.")


class TodoUpdateTool(ToolSpec):
    def __init__(self, name: str = "todo_update") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Update the status of a todo item by index."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Zero-based index of the item"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            },
            "required": ["index", "status"],
        }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        index = input.get("index", -1)
        status = input.get("status", "pending")
        _global_todos.update(index, status)
        return ToolResult.success(f"Item {index} updated to {status}.")


class TodoListTool(ToolSpec):
    def __init__(self, name: str = "todo_list") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "List all todo items."

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        items = _global_todos.list()
        if not items:
            return ToolResult.success("No todo items.")
        lines = []
        for i, item in enumerate(items):
            lines.append(f"{i}. [{item.get('status', 'pending')}] {item.get('content', '')}")
        return ToolResult.success("\n".join(lines))


class UpdatePlanTool(ToolSpec):
    name = "update_plan"
    description = "Update the implementation plan with steps and their status."
    input_schema = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["step", "status"],
                },
            },
            "explanation": {"type": "string", "description": "Optional high-level explanation"},
        },
        "required": ["plan"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        plan = input.get("plan", [])
        explanation = input.get("explanation", "")
        _global_plan.update(plan)
        result = f"Plan updated with {len(plan)} steps."
        if explanation:
            result += f"\n{explanation}"
        return ToolResult.success(result)


class DiagnosticsTool(ToolSpec):
    name = "diagnostics"
    description = "Report workspace info, git detection, and environment status."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        import os, platform, subprocess
        lines = []
        lines.append(f"Workspace: {context.params.workspace.resolve()}")
        lines.append(f"CWD: {context.params.cwd.resolve()}")
        lines.append(f"Python: {platform.python_version()}")
        lines.append(f"Platform: {platform.system()} {platform.release()}")
        # Git detection
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
                cwd=context.params.workspace,
            )
            if result.returncode == 0:
                lines.append(f"Git root: {result.stdout.strip()}")
            else:
                lines.append("Git: not detected")
        except Exception:
            lines.append("Git: not available")
        # rg/fd detection
        for cmd in ["rg", "fd", "git"]:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
                lines.append(f"{cmd}: available")
            except Exception:
                lines.append(f"{cmd}: not available")
        return ToolResult.success("\n".join(lines))
