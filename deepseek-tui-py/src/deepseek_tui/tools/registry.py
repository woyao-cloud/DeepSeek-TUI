"""ToolRegistry — register, discover, and execute tools.

Port of `crates/tui/src/tools/registry.rs`.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ToolSpec, ToolResult, ToolError


class ToolRegistry:
    """Registry that holds all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        name = tool.name
        self._tools[name] = tool

    def register_all(self, tools: list[ToolSpec]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def contains(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)

    async def execute(self, name: str, input: dict[str, Any], context) -> str:
        tool = self.get(name)
        if tool is None:
            raise ToolError.not_available(f"tool '{name}' is not registered")
        result = await tool.execute(input, context)
        return result.content

    async def execute_full(self, name: str, input: dict[str, Any], context) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            raise ToolError.not_available(f"tool '{name}' is not registered")
        return await tool.execute(input, context)

    def to_api_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to API format (sorted by name for cache stability)."""
        tools = sorted(self._tools.values(), key=lambda t: t.name)
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def read_only_tools(self) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.is_read_only()]

    def mutating_tools(self) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.is_mutating()]


class ToolRegistryBuilder:
    """Builder for constructing a ToolRegistry with common tool sets."""

    def __init__(self) -> None:
        self._tools: list[ToolSpec] = []

    def with_tool(self, tool: ToolSpec) -> ToolRegistryBuilder:
        self._tools.append(tool)
        return self

    def build(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_all(self._tools)
        return registry
