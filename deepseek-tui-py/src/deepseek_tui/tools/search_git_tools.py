"""Search and Git tools — grep_files, file_search, git_status, git_diff, etc.

Port of `crates/tui/src/tools/search.rs`, `file_search.rs`, `git.rs`, `git_history.rs`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .base import ToolSpec, ToolResult, ToolCapability
from .context import ToolContext


# ── Search Tools ──────────────────────────────────────────────────────

class GrepFilesTool(ToolSpec):
    name = "grep_files"
    description = "Search for a regex pattern in files within the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression pattern"},
            "path": {"type": "string", "description": "Directory to search (default: workspace)"},
            "include": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns to include"},
            "context_lines": {"type": "integer", "description": "Lines of context before/after match"},
            "max_results": {"type": "integer", "description": "Maximum results (default: 100)"},
        },
        "required": ["pattern"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input.get("pattern", "")
        search_path = input.get("path", str(context.params.workspace))
        include = input.get("include", None)
        context_lines = input.get("context_lines", 2)
        max_results = input.get("max_results", 100)
        if not pattern:
            return ToolResult.error("Missing required field 'pattern'")
        try:
            cmd = ["rg", "-n", "--no-heading"]
            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])
            cmd.extend(["--max-count", str(max_results)])
            if include:
                for g in include:
                    cmd.extend(["-g", g])
            cmd.append(pattern)
            cmd.append(search_path)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return ToolResult.success(result.stdout.rstrip())
            elif result.returncode == 1:
                return ToolResult.success("No matches found.")
            else:
                # Fall back to grep if ripgrep not available
                return await self._grep_fallback(pattern, search_path, include, context_lines, max_results)
        except FileNotFoundError:
            return await self._grep_fallback(pattern, search_path, include, context_lines, max_results)
        except subprocess.TimeoutExpired:
            return ToolResult.error("Search timed out")
        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")

    async def _grep_fallback(self, pattern, search_path, include, context_lines, max_results):
        try:
            cmd = ["grep", "-rn", "--no-messages"]
            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])
            cmd.append(pattern)
            cmd.append(search_path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            lines = result.stdout.strip().split("\n")[:max_results]
            return ToolResult.success("\n".join(lines) if lines else "No matches found.")
        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")


class FileSearchTool(ToolSpec):
    name = "file_search"
    description = "Search for files using fuzzy matching."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "File name or path fragment"},
            "path": {"type": "string", "description": "Base path to search"},
            "limit": {"type": "integer", "description": "Maximum results (default: 20)"},
        },
        "required": ["query"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = input.get("query", "")
        search_path = input.get("path", str(context.params.workspace))
        limit = input.get("limit", 20)
        if not query:
            return ToolResult.error("Missing required field 'query'")
        try:
            cmd = ["fd", "--max-results", str(limit), query, search_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                return ToolResult.success(result.stdout.rstrip())
            else:
                return await self._find_fallback(query, search_path, limit)
        except FileNotFoundError:
            return await self._find_fallback(query, search_path, limit)
        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")

    async def _find_fallback(self, query, search_path, limit):
        try:
            cmd = ["find", search_path, "-iname", f"*{query}*", "-type", "f"]
            if limit > 0:
                cmd.extend(["-maxdepth", "5"])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            lines = result.stdout.strip().split("\n")[:limit]
            return ToolResult.success("\n".join(lines) if lines else "No matches found.")
        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")


# ── Git Tools ─────────────────────────────────────────────────────────

def _git_cmd(args: list[str], cwd: Path) -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, timeout=30, cwd=cwd
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        raise ToolError.not_available("git not found")
    except subprocess.TimeoutExpired:
        raise ToolError.execution_failed("git command timed out")


class GitStatusTool(ToolSpec):
    name = "git_status"
    description = "Show the working tree status (git status)."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Optional repo path"}},
        "required": [],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        cwd = Path(input.get("path", str(context.params.workspace)))
        try:
            stdout, stderr, rc = _git_cmd(["status"], cwd)
            return ToolResult.success(stdout or stderr)
        except Exception as e:
            return ToolResult.error(str(e))


class GitDiffTool(ToolSpec):
    name = "git_diff"
    description = "Show changes in the working tree or between commits (git diff)."
    input_schema = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean", "description": "Show staged changes only"},
            "base": {"type": "string", "description": "Base ref to diff against"},
        },
        "required": [],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        cwd = context.params.workspace
        staged = input.get("staged", False)
        base = input.get("base", None)
        try:
            args = ["diff"]
            if staged:
                args.append("--cached")
            if base:
                args.append(base)
            else:
                args.append("HEAD")
            stdout, stderr, rc = _git_cmd(args, cwd)
            return ToolResult.success(stdout or "(no output)")
        except Exception as e:
            return ToolResult.error(str(e))


class GitLogTool(ToolSpec):
    name = "git_log"
    description = "Show commit history (git log)."
    input_schema = {
        "type": "object",
        "properties": {
            "max_count": {"type": "integer", "description": "Maximum commits (default: 10)"},
        },
        "required": [],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        cwd = context.params.workspace
        max_count = input.get("max_count", 10)
        try:
            stdout, stderr, rc = _git_cmd(
                ["log", f"--max-count={max_count}", "--oneline", "--decorate"], cwd
            )
            return ToolResult.success(stdout or "(no commits)")
        except Exception as e:
            return ToolResult.error(str(e))


class GitShowTool(ToolSpec):
    name = "git_show"
    description = "Show the details of a specific commit (git show)."
    input_schema = {
        "type": "object",
        "properties": {
            "revision": {"type": "string", "description": "Revision (commit hash, tag, etc.)"},
        },
        "required": ["revision"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        cwd = context.params.workspace
        revision = input.get("revision", "")
        if not revision:
            return ToolResult.error("Missing required field 'revision'")
        try:
            stdout, stderr, rc = _git_cmd(["show", revision], cwd)
            return ToolResult.success(stdout or stderr)
        except Exception as e:
            return ToolResult.error(str(e))


class GitBlameTool(ToolSpec):
    name = "git_blame"
    description = "Show who last modified each line of a file (git blame)."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to blame"},
        },
        "required": ["path"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        cwd = context.params.workspace
        path = input.get("path", "")
        if not path:
            return ToolResult.error("Missing required field 'path'")
        try:
            stdout, stderr, rc = _git_cmd(["blame", path], cwd)
            return ToolResult.success(stdout or stderr)
        except Exception as e:
            return ToolResult.error(str(e))
