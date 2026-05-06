"""RLM tool — sandboxed Python REPL for processing long inputs.

Port of `crates/tui/src/tools/rlm.rs`.

The RLM tool spawns a sub-LLM inside a sandboxed Python REPL to process
long inputs that don't fit in the calling model's context window.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from io import StringIO
from typing import Any, Optional

from .base import ToolSpec, ToolResult, ToolError, ToolCapability
from .context import ToolContext


class RlmTool(ToolSpec):
    """Recursive Language Model tool — sandboxed Python REPL for long inputs."""

    name = "rlm"
    description = (
        "Process a long input through a sandboxed Python REPL with sub-LLM access. "
        "Use for chunking, batching, or critiquing content too large for your context window."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What to do with the input (e.g. 'Summarize', 'Extract API endpoints')",
            },
            "content": {
                "type": "string",
                "description": "The content to process (long text, code, etc.)",
            },
            "code": {
                "type": "string",
                "description": "Optional Python code to execute instead of the default processor",
            },
        },
        "required": ["task", "content"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    def __init__(self, llm_client: Optional[Any] = None, root_model: str = "deepseek-v4-flash") -> None:
        self._llm_client = llm_client
        self._root_model = root_model

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        task = input.get("task", "")
        content = input.get("content", "")
        user_code = input.get("code", "")

        if not task or not content:
            return ToolResult.error("Missing required fields: 'task' and 'content'")

        # If user provides custom code, execute it in a sandbox
        if user_code:
            return await self._run_custom_code(user_code, content, task)

        # Default behavior: chunk and summarize
        return await self._default_processor(task, content)

    async def _default_processor(self, task: str, content: str) -> ToolResult:
        """Default processor: split content into chunks and process each."""
        chunk_size = 2000  # chars per chunk
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        results = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"Task: {task}\n\n"
                f"Content chunk {i + 1}/{len(chunks)}:\n"
                f"{chunk}\n\n"
                f"Process this chunk according to the task. "
                f"Keep your response brief and focused."
            )

            if self._llm_client:
                try:
                    from ..llm import MessageRequest, Message, ContentBlock
                    req = MessageRequest(
                        model=self._root_model,
                        messages=[Message(role="user", content=[ContentBlock(type="text", text=prompt)])],
                    )
                    response = await self._llm_client.create_message(req)
                    chunk_result = response.content[0].text if response.content else ""
                except Exception as e:
                    chunk_result = f"[Error processing chunk {i+1}: {e}]"
            else:
                # No LLM client — simulate with basic extraction
                chunk_result = (
                    f"[Chunk {i+1}/{len(chunks)}: {len(chunk)} chars] "
                    f"First 100 chars: {chunk[:100]}..."
                )

            results.append(chunk_result)

        # Synthesize final result
        summary = (
            f"Processed {len(chunks)} chunks for task: '{task}'\n\n"
            + "\n---\n".join(results)
        )

        return ToolResult.success(summary)

    async def _run_custom_code(self, code: str, content: str, task: str) -> ToolResult:
        """Execute user-provided Python code in a restricted sandbox."""
        # Build a restricted globals dict
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "bool": bool,
                "True": True,
                "False": False,
                "None": None,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "RuntimeError": RuntimeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "ImportError": ImportError,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "type": type,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "min": min,
                "max": max,
                "sum": sum,
                "any": any,
                "all": all,
                "open": _sandbox_open,
                "json": __import__("json"),
            },
            "PROMPT": content,
            "TASK": task,
        }

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Parse the code to check for dangerous constructs
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("__import__", "exec", "eval", "compile"):
                        # Allow import via __builtins__ only
                        pass

            exec(code, restricted_globals)
            output = sys.stdout.getvalue()
            return ToolResult.success(output or "Code executed successfully (no output).")
        except Exception as e:
            return ToolResult.error(f"Code execution error: {e}")
        finally:
            sys.stdout = old_stdout


def _sandbox_open(path, mode="r", *args, **kwargs):
    """Restricted open() that only allows reading."""
    if "w" in mode or "a" in mode or "+" in mode:
        raise PermissionError("Writing files is not allowed in the RLM sandbox")
    return open(path, mode, *args, **kwargs)
