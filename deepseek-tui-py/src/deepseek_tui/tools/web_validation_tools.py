"""Web and validation tools — web_search, fetch_url, validate_data, request_user_input.

Port of `crates/tui/src/tools/web_search.rs`, `fetch_url.rs`, `validate_data.rs`, `user_input.rs`.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import httpx

from .base import ToolSpec, ToolResult, ToolCapability, ApprovalRequirement
from .context import ToolContext


class WebSearchTool(ToolSpec):
    name = "web_search"
    description = "Search the web for information. Returns search results with snippets."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }

    def capabilities(self):
        return [ToolCapability.NETWORK]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = input.get("query", "")
        if not query:
            return ToolResult.error("Missing required field 'query'")
        # Use DuckDuckGo via httpx (no API key needed)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for topic in data.get("RelatedTopics", []):
                        if "Text" in topic and "FirstURL" in topic:
                            results.append(f"- {topic['Text']}\n  {topic['FirstURL']}")
                            if len(results) >= 8:
                                break
                    if results:
                        return ToolResult.success("\n\n".join(results))
                    return ToolResult.success(f"No results for '{query}'.")
                return ToolResult.error(f"Search failed: HTTP {resp.status_code}")
        except Exception as e:
            return ToolResult.error(f"Search failed: {e}")


class FetchUrlTool(ToolSpec):
    name = "fetch_url"
    description = "Fetch a URL and return its contents as text."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
        },
        "required": ["url"],
    }

    def capabilities(self):
        return [ToolCapability.NETWORK]

    def approval_requirement(self):
        return ApprovalRequirement.SUGGEST

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = input.get("url", "")
        if not url:
            return ToolResult.error("Missing required field 'url'")
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "text" in content_type or "json" in content_type or "xml" in content_type:
                    text = resp.text
                else:
                    text = f"[Binary content: {len(resp.content)} bytes, {content_type}]"
                return ToolResult.success(f"URL: {url}\n\n{text[:20000]}")
        except httpx.HTTPStatusError as e:
            return ToolResult.error(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            return ToolResult.error(f"Failed to fetch URL: {e}")


class ValidateDataTool(ToolSpec):
    name = "validate_data"
    description = "Validate JSON or TOML data against a schema or parse it."
    input_schema = {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "Data to validate (JSON or TOML)"},
            "format": {"type": "string", "enum": ["json", "toml"], "description": "Data format"},
        },
        "required": ["data"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        data = input.get("data", "")
        fmt = input.get("format", "json")
        if not data:
            return ToolResult.error("Missing required field 'data'")
        try:
            if fmt == "json":
                parsed = json.loads(data)
                return ToolResult.success(json.dumps(parsed, indent=2, ensure_ascii=False))
            elif fmt == "toml":
                import tomllib
                parsed = tomllib.loads(data)
                return ToolResult.success(json.dumps(parsed, indent=2, ensure_ascii=False))
            else:
                return ToolResult.error(f"Unsupported format: {fmt}")
        except json.JSONDecodeError as e:
            return ToolResult.error(f"Invalid JSON: {e}")
        except Exception as e:
            return ToolResult.error(f"Validation failed: {e}")


class RequestUserInputTool(ToolSpec):
    name = "request_user_input"
    description = "Ask the user a question and return their response."
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to ask the user"},
        },
        "required": ["question"],
    }

    def capabilities(self):
        return [ToolCapability.READ_ONLY]

    async def execute(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        question = input.get("question", "")
        if not question:
            return ToolResult.error("Missing required field 'question'")
        # In a real TUI this would show a dialog; for CLI mode we read from stdin
        import sys
        print(f"\n[User Input Required] {question}", file=sys.stderr)
        print("> ", end="", flush=True, file=sys.stderr)
        answer = sys.stdin.readline().strip()
        return ToolResult.success(f"User responded: {answer}")
