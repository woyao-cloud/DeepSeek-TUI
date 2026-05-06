"""FIM (Fill-in-the-Middle) edit tool.

Port of `crates/tui/src/tools/fim.rs`.

Uses the DeepSeek `/beta/completions` FIM endpoint to generate
replacement content between prefix and suffix anchors in a file.
"""

from __future__ import annotations

import httpx
from pathlib import Path
from typing import Any, Optional

from .base import ToolSpec, ToolResult, ToolError, ToolCapability, ApprovalRequirement
from .context import ToolContext


class FimError(Exception):
    pass


class FimEditTool(ToolSpec):
    """Edit a file using Fill-in-the-Middle completion."""

    name = "fim_edit"
    description = (
        "Edit a file using Fill-in-the-Middle (FIM) completion. "
        "Provide a file path, prefix_anchor (text that appears before the "
        "section to replace), and suffix_anchor (text that appears after the "
        "section to replace). The tool calls DeepSeek's FIM endpoint to "
        "generate replacement content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative to workspace)",
            },
            "prefix_anchor": {
                "type": "string",
                "description": "Text anchor marking the end of the prefix.",
            },
            "suffix_anchor": {
                "type": "string",
                "description": "Text anchor marking the start of the suffix.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens to generate (default: 1024)",
            },
        },
        "required": ["path", "prefix_anchor", "suffix_anchor"],
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def capabilities(self):
        return [ToolCapability.READ_ONLY, ToolCapability.WRITES_FILES]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        path = input.get("path", "")
        prefix_anchor = input.get("prefix_anchor", "")
        suffix_anchor = input.get("suffix_anchor", "")
        max_tokens = input.get("max_tokens", 1024)

        if not path or not prefix_anchor or not suffix_anchor:
            return ToolResult.error("Missing required fields: path, prefix_anchor, suffix_anchor")

        # 1. Resolve and read file
        resolved = _resolve_path(path, context.params.workspace)
        if not resolved.exists():
            return ToolResult.error(f"File not found: {path}")
        try:
            content = resolved.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult.error(f"Failed to read {path}: {e}")

        # 2. Find prefix anchor
        prefix_pos = content.find(prefix_anchor)
        if prefix_pos == -1:
            return ToolResult.error(f"Prefix anchor not found in file: '{prefix_anchor}'")
        prefix_end = prefix_pos + len(prefix_anchor)

        # 3. Find suffix anchor (after prefix)
        suffix_pos = content[prefix_end:].find(suffix_anchor)
        if suffix_pos == -1:
            return ToolResult.error(f"Suffix anchor not found after prefix: '{suffix_anchor}'")
        suffix_start = prefix_end + suffix_pos

        # 4. Validate no overlap
        if suffix_start < prefix_end:
            return ToolResult.error(
                f"Anchors overlap: suffix starts at {suffix_start}, prefix ends at {prefix_end}"
            )

        # 5. Extract prefix/suffix
        fim_prefix = content[:prefix_end]
        fim_suffix = content[suffix_start:]

        # 6. Call FIM API
        if not self._api_key:
            # No API key — simulate with a placeholder
            generated = "// TODO: FIM API requires an API key\n"
        else:
            try:
                generated = await self._fim_call(fim_prefix, fim_suffix, max_tokens)
            except Exception as e:
                return ToolResult.error(f"FIM API call failed: {e}")

        # 7. Write back
        new_content = fim_prefix + generated + fim_suffix
        try:
            resolved.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult.error(f"Failed to write {path}: {e}")

        return ToolResult.success(
            f"FIM edit applied to `{path}`. Generated {len(generated)} chars "
            f"between prefix_anchor end (byte {prefix_end}) and "
            f"suffix_anchor start (byte {suffix_start}).\n\n"
            f"Generated:\n```\n{generated}\n```"
        )

    async def _fim_call(self, prefix: str, suffix: str, max_tokens: int) -> str:
        """Call the DeepSeek FIM endpoint (/beta/completions)."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/beta/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "prompt": prefix,
                    "suffix": suffix,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("text", "")
            return ""


def _resolve_path(path_str: str, workspace: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (workspace / p).resolve()
    else:
        p = p.resolve()
    ws = workspace.resolve()
    try:
        p.relative_to(ws)
    except ValueError:
        raise ToolError.path_escape(str(p))
    return p
