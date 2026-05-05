"""Shell tools — exec_shell, exec_shell_wait, exec_shell_interact, note.

Port of `crates/tui/src/tools/shell.rs`.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .base import ToolSpec, ToolResult, ToolError, ToolCapability, ApprovalRequirement
from .context import ToolContext


# ── In-memory background task store (simplified) ─────────────────────

_background_tasks: dict[str, subprocess.Popen] = {}


class ExecShellTool(ToolSpec):
    name = "exec_shell"
    description = "Execute a shell command in the workspace directory."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default: 120000)"},
            "background": {"type": "boolean", "description": "Run in background (default: false)"},
            "cwd": {"type": "string", "description": "Working directory (default: workspace)"},
        },
        "required": ["command"],
    }

    def capabilities(self):
        return [ToolCapability.EXECUTES_CODE]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        command = input.get("command", "")
        if not command:
            return ToolResult.error("Missing required field 'command'")
        timeout_ms = input.get("timeout_ms", 120000)
        background = input.get("background", False)
        cwd_str = input.get("cwd", str(context.params.workspace))
        cwd = Path(cwd_str).resolve()

        if background:
            return await self._run_background(command, cwd)
        return await self._run_foreground(command, cwd, timeout_ms)

    async def _run_foreground(self, command: str, cwd: Path, timeout_ms: int) -> ToolResult:
        timeout_s = timeout_ms / 1000.0
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult.error(f"Command timed out after {timeout_ms}ms")
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            output = out
            if err:
                output += f"\nSTDERR:\n{err}"
            if proc.returncode == 0:
                return ToolResult.success(output)
            else:
                return ToolResult.error(f"Exit code {proc.returncode}:\n{output}")
        except FileNotFoundError:
            return ToolResult.error(f"Command not found: {command.split()[0]}")
        except Exception as e:
            return ToolResult.error(f"Execution failed: {e}")

    async def _run_background(self, command: str, cwd: Path) -> ToolResult:
        task_id = f"shell-{id(command)}-{len(_background_tasks)}"
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            _background_tasks[task_id] = proc
            return ToolResult.success(f"Started background task {task_id} (PID {proc.pid})")
        except Exception as e:
            return ToolResult.error(f"Failed to start background task: {e}")


class ShellWaitTool(ToolSpec):
    """Wait for a background shell task to complete."""

    def __init__(self, name: str = "exec_shell_wait") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Wait for a background shell task and return its output."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID from exec_shell"},
                "timeout_ms": {"type": "integer", "description": "Max wait time in ms"},
            },
            "required": ["task_id"],
        }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = input.get("task_id", "")
        timeout_ms = input.get("timeout_ms", 30000)
        proc = _background_tasks.get(task_id)
        if proc is None:
            return ToolResult.error(f"Background task not found: {task_id}")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_ms / 1000.0
            )
            del _background_tasks[task_id]
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            output = out
            if err:
                output += f"\nSTDERR:\n{err}"
            status = "completed" if proc.returncode == 0 else f"failed (code {proc.returncode})"
            return ToolResult.success(f"Task {task_id} {status}:\n{output}")
        except asyncio.TimeoutError:
            return ToolResult.success(f"Task {task_id} still running (timeout)")


class ShellInteractTool(ToolSpec):
    """Send input to a running background shell task."""

    def __init__(self, name: str = "exec_shell_interact") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Send input to a running background shell task."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "input": {"type": "string", "description": "Input to send to stdin"},
            },
            "required": ["task_id", "input"],
        }

    def capabilities(self):
        return [ToolCapability.EXECUTES_CODE]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = input.get("task_id", "")
        stdin_data = input.get("input", "")
        proc = _background_tasks.get(task_id)
        if proc is None:
            return ToolResult.error(f"Background task not found: {task_id}")
        if proc.stdin is None:
            return ToolResult.error("Task has no stdin (already completed)")
        try:
            proc.stdin.write((stdin_data + "\n").encode())
            proc.stdin.flush()
            return ToolResult.success(f"Sent input to task {task_id}")
        except Exception as e:
            return ToolResult.error(f"Failed to send input: {e}")


class NoteTool(ToolSpec):
    name = "note"
    description = "Save a persistent note visible across the session."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Note content"},
        },
        "required": ["content"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        content = input.get("content", "")
        if not content:
            return ToolResult.error("Missing required field 'content'")
        return ToolResult.success(f"Note saved: {content[:80]}...")
