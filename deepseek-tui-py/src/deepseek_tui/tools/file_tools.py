"""File tools — read_file, write_file, edit_file, list_dir.

Port of `crates/tui/src/tools/file.rs`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import ToolSpec, ToolResult, ToolError, ToolCapability, ApprovalRequirement
from .context import ToolContext


def _resolve_path(path_str: str, workspace: Path) -> Path:
    """Resolve a path, ensuring it doesn't escape the workspace."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (workspace / p).resolve()
    else:
        p = p.resolve()
    # Normalize for comparison
    ws = workspace.resolve()
    try:
        p.relative_to(ws)
    except ValueError:
        raise ToolError.path_escape(str(p))
    return p


class ReadFileTool(ToolSpec):
    name = "read_file"
    description = "Read the contents of a file from the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file (relative or absolute)"},
        },
        "required": ["path"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = input.get("path", "")
        if not path_str:
            return ToolResult.error("Missing required field 'path'")
        try:
            path = _resolve_path(path_str, context.params.workspace)
            if not path.exists():
                return ToolResult.error(f"File not found: {path}")
            if not path.is_file():
                return ToolResult.error(f"Not a file: {path}")
            content = path.read_text(encoding="utf-8")
            return ToolResult.success(content)
        except ToolError as e:
            return ToolResult.error(str(e))
        except Exception as e:
            return ToolResult.error(f"Failed to read file: {e}")


class WriteFileTool(ToolSpec):
    name = "write_file"
    description = "Write content to a file in the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def capabilities(self):
        return [ToolCapability.WRITES_FILES]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = input.get("path", "")
        content = input.get("content", "")
        if not path_str:
            return ToolResult.error("Missing required field 'path'")
        path = _resolve_path(path_str, context.params.workspace)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult.success(f"Written {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult.error(f"Failed to write file: {e}")


class EditFileTool(ToolSpec):
    name = "edit_file"
    description = "Replace text in a file using search/replace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "search": {"type": "string", "description": "Text to search for"},
            "replace": {"type": "string", "description": "Text to replace with"},
        },
        "required": ["path", "search", "replace"],
    }

    def capabilities(self):
        return [ToolCapability.WRITES_FILES]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = input.get("path", "")
        search = input.get("search", "")
        replace = input.get("replace", "")
        if not path_str or not search:
            return ToolResult.error("Missing required fields: 'path' and 'search'")
        path = _resolve_path(path_str, context.params.workspace)
        if not path.exists():
            return ToolResult.error(f"File not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
            if search not in content:
                return ToolResult.error(f"Search text not found in {path}")
            new_content = content.replace(search, replace, 1)
            path.write_text(new_content, encoding="utf-8")
            return ToolResult.success(f"Replaced text in {path}")
        except Exception as e:
            return ToolResult.error(f"Failed to edit file: {e}")


class ListDirTool(ToolSpec):
    name = "list_dir"
    description = "List entries in a directory."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default: workspace root)"},
        },
        "required": [],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = input.get("path", ".")
        path = _resolve_path(path_str, context.params.workspace)
        if not path.exists():
            return ToolResult.error(f"Directory not found: {path}")
        if not path.is_dir():
            return ToolResult.error(f"Not a directory: {path}")
        try:
            entries = []
            for entry in sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
                suffix = "/" if entry.is_dir() else ""
                entries.append(f"{entry.name}{suffix}")
            return ToolResult.success("\n".join(entries))
        except Exception as e:
            return ToolResult.error(f"Failed to list directory: {e}")


class ApplyPatchTool(ToolSpec):
    name = "apply_patch"
    description = "Apply a unified diff patch to the working tree."
    input_schema = {
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "Unified diff patch content"},
        },
        "required": ["patch"],
    }

    def capabilities(self):
        return [ToolCapability.WRITES_FILES]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        patch = input.get("patch", "")
        if not patch:
            return ToolResult.error("Missing required field 'patch'")
        # Write patch to temp file and apply with git apply
        import tempfile, subprocess
        cwd = context.params.workspace
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(patch)
                patch_path = f.name
            result = subprocess.run(
                ["git", "apply", "--recount", patch_path],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            os.unlink(patch_path)
            if result.returncode == 0:
                return ToolResult.success("Patch applied successfully")
            else:
                return ToolResult.error(f"Patch failed: {result.stderr.strip()}")
        except FileNotFoundError:
            return ToolResult.error("git not found — is it installed?")
        except subprocess.TimeoutExpired:
            return ToolResult.error("Patch application timed out")
        except Exception as e:
            return ToolResult.error(f"Failed to apply patch: {e}")
