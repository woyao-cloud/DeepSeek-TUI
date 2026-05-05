"""MCP client — port of `deepseek-mcp` crate.

Minimal MCP client for connecting to stdio / SSE tool servers.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class McpToolDef:
    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpServerConfig:
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    url: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    disabled: bool = False


@dataclass
class McpConnection:
    server_name: str
    process: Optional[subprocess.Popen] = None
    tools: list[McpToolDef] = field(default_factory=list)
    ready: bool = False
    error: Optional[str] = None


class McpManager:
    """Manages MCP server connections and tool discovery."""

    def __init__(self) -> None:
        self._servers: dict[str, McpServerConfig] = {}
        self._connections: dict[str, McpConnection] = {}

    def register(self, name: str, config: McpServerConfig) -> None:
        self._servers[name] = config

    def register_all(self, configs: dict[str, McpServerConfig]) -> None:
        self._servers.update(configs)

    async def start_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, config in self._servers.items():
            if config.disabled:
                results[name] = False
                continue
            try:
                proc = subprocess.Popen(
                    [config.command] + config.args if config.command else [],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**__import__("os").environ, **config.env},
                )
                self._connections[name] = McpConnection(
                    server_name=name,
                    process=proc,
                    ready=True,
                )
                results[name] = True
            except Exception as e:
                self._connections[name] = McpConnection(
                    server_name=name,
                    error=str(e),
                    ready=False,
                )
                results[name] = False
        return results

    def get_tools(self, server_name: str) -> list[McpToolDef]:
        conn = self._connections.get(server_name)
        if conn is None or not conn.ready:
            return []
        return conn.tools

    def all_tools(self) -> dict[str, McpToolDef]:
        result: dict[str, McpToolDef] = {}
        for conn in self._connections.values():
            if conn.ready:
                for tool in conn.tools:
                    result[f"{conn.server_name}:{tool.name}"] = tool
        return result

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        conn = self._connections.get(server_name)
        if conn is None or not conn.ready or conn.process is None:
            raise RuntimeError(f"MCP server '{server_name}' not ready")
        # Simple JSON-RPC call over stdio
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        if conn.process.stdin:
            conn.process.stdin.write(json.dumps(request) + "\n")
            conn.process.stdin.flush()
        if conn.process.stdout:
            line = conn.process.stdout.readline()
            if line:
                return json.loads(line)
        return {"error": "no response"}
