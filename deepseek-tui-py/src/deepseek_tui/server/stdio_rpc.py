"""JSON-RPC over stdio — port of `deepseek-app-server` stdio transport.

Provides a line-delimited JSON-RPC 2.0 server over stdin/stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from ..core import Runtime
from ..protocol import ThreadRequest, ThreadResponse, AppRequest, AppResponse


class StdioRpcServer:
    """JSON-RPC 2.0 server over stdin/stdout."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._running = True

    async def run(self) -> None:
        """Read JSON-RPC requests from stdin, write responses to stdout."""
        while self._running:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
                continue

            request_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params", {})

            if request.get("jsonrpc") != "2.0":
                self._send_error(request_id, -32600, "Invalid Request: jsonrpc must be 2.0")
                continue

            try:
                await self._dispatch(method, params, request_id)
            except Exception as e:
                self._send_error(request_id, -32603, f"Internal error: {e}")

    async def _dispatch(self, method: str, params: Any, request_id: Any) -> None:
        match method:
            case "healthz" | "app/healthz":
                self._send_result(request_id, {"status": "ok", "service": "deepseek-api"})

            case "capabilities":
                self._send_result(request_id, {
                    "transport": "stdio",
                    "methods": [
                        "healthz", "capabilities",
                        "thread/request", "thread/create", "thread/start",
                        "thread/list", "thread/read",
                        "prompt/request", "prompt/run",
                        "app/request", "app/capabilities",
                        "shutdown",
                    ],
                })

            case "thread/request" | "thread/create" | "thread/start" | "thread/list" | "thread/read":
                await self._handle_thread(method, params, request_id)

            case "prompt/request" | "prompt/run":
                await self._handle_prompt(params, request_id)

            case "app/request" | "app/capabilities":
                await self._handle_app(method, params, request_id)

            case "shutdown":
                self._running = False
                self._send_result(request_id, {"ok": True, "status": "stopped"})

            case _:
                self._send_error(request_id, -32601, f"Method not found: {method}")

    async def _handle_thread(self, method: str, params: Any, request_id: Any) -> None:
        kind_map = {
            "thread/create": "create",
            "thread/start": "start",
            "thread/list": "list",
            "thread/read": "read",
            "thread/request": params.get("kind", "message") if isinstance(params, dict) else "message",
        }
        kind = kind_map.get(method, "message")
        req = ThreadRequest(kind=kind, **(params if isinstance(params, dict) else {}))
        try:
            response = await self._runtime.handle_thread(req)
            self._send_result(request_id, response.model_dump())
        except Exception as e:
            self._send_error(request_id, -32603, str(e))

    async def _handle_prompt(self, params: Any, request_id: Any) -> None:
        from ..protocol import PromptRequest

        if isinstance(params, dict):
            req = PromptRequest(prompt=params.get("prompt", ""), model=params.get("model"))
            try:
                response = await self._runtime.handle_prompt(req)
                self._send_result(request_id, response.model_dump())
            except Exception as e:
                self._send_error(request_id, -32603, str(e))
        else:
            self._send_error(request_id, -32602, "Invalid params")

    async def _handle_app(self, method: str, params: Any, request_id: Any) -> None:
        if method == "app/capabilities":
            self._send_result(request_id, {
                "routes": ["thread", "prompt", "app"],
                "config": ["get", "set", "list"],
            })
            return

        if isinstance(params, dict):
            kind = params.get("kind", "")
            if kind == "models":
                from ..agent import ModelRegistry
                registry = ModelRegistry()
                self._send_result(request_id, {"models": [m.id for m in registry.list()]})
                return

        self._send_result(request_id, {"ok": True})

    def _send_result(self, request_id: Any, result: Any) -> None:
        response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
