"""LSP client — stdio transport for JSON-RPC LSP communication.

Port of `crates/tui/src/lsp/client.rs`.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from .models import Diagnostic, Severity
from .registry import Language


class LspTransport:
    """Abstract LSP transport interface."""

    async def diagnostics_for(
        self, path: Path, text: str, wait_ms: int = 5000
    ) -> list[Diagnostic]:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError


class StdioLspTransport(LspTransport):
    """LSP transport over stdio (JSON-RPC)."""

    def __init__(
        self, lang: Language, workspace: Path,
        process: asyncio.subprocess.Process,
    ) -> None:
        self._lang = lang
        self._workspace = workspace
        self._process = process
        self._seq = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._initialized = False

    @classmethod
    async def spawn(
        cls, command: str, args: list[str], lang: Language, workspace: Path,
    ) -> StdioLspTransport:
        """Spawn an LSP server process."""
        proc = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=workspace,
        )
        transport = cls(lang, workspace, proc)
        transport._reader_task = asyncio.create_task(transport._read_loop())
        await transport._initialize()
        return transport

    async def _initialize(self) -> None:
        """Send LSP initialize request + initialized notification."""
        result = await self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": self._workspace.as_uri(),
            "capabilities": {
                "textDocument": {
                    "diagnostics": True,
                    "didChange": True,
                },
            },
        })
        self._initialized = True
        await self._notify("initialized", {})

    async def _send(self, method: str, params: Any, msg_id: Optional[int] = None) -> None:
        """Send a JSON-RPC message."""
        msg: dict[str, Any] = {"jsonrpc": "2.0"}
        if msg_id is not None:
            msg["id"] = msg_id
            msg["method"] = method
            msg["params"] = params
        else:
            msg["method"] = method
            msg["params"] = params
        body = json.dumps(msg)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        if self._process.stdin:
            self._process.stdin.write((header + body).encode())
            await self._process.stdin.drain()

    async def _request(self, method: str, params: Any) -> Any:
        """Send a request and wait for response."""
        self._seq += 1
        msg_id = self._seq
        future = asyncio.get_event_loop().create_future()
        self._pending[str(msg_id)] = future
        await self._send(method, params, msg_id)
        try:
            return await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            self._pending.pop(str(msg_id), None)
            raise

    async def _notify(self, method: str, params: Any) -> None:
        """Send a notification (no response expected)."""
        await self._send(method, params)

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from the server's stdout."""
        buffer = ""
        while True:
            chunk = await self._process.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            # Process complete messages
            while "\r\n\r\n" in buffer:
                header, rest = buffer.split("\r\n\r\n", 1)
                if "Content-Length:" in header:
                    length = int(header.split("Content-Length:")[1].strip())
                    if len(rest) >= length:
                        msg = json.loads(rest[:length])
                        buffer = rest[length:]
                        await self._handle_message(msg)
                    else:
                        break
                else:
                    break

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Handle an incoming JSON-RPC message."""
        msg_id = msg.get("id")
        if msg_id is not None:
            future = self._pending.pop(str(msg_id), None)
            if future and not future.done():
                if "error" in msg:
                    future.set_exception(Exception(msg["error"]))
                else:
                    future.set_result(msg.get("result"))
        # PublishDiagnostics notifications
        if msg.get("method") == "textDocument/publishDiagnostics":
            pass  # We'll handle these through the diagnostics_for path

    async def diagnostics_for(
        self, path: Path, text: str, wait_ms: int = 5000
    ) -> list[Diagnostic]:
        """Request diagnostics by opening a file and waiting."""
        uri = path.as_uri()
        # didOpen
        await self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": self._lang.value,
                "version": 1,
                "text": text,
            },
        })

        # Poll for publishDiagnostics (simplified: wait, then didClose)
        await asyncio.sleep(wait_ms / 1000.0)

        # didClose
        await self._notify("textDocument/didClose", {
            "textDocument": {"uri": uri},
        })

        return []  # Full pub/sub implementation would capture diagnostics

    async def shutdown(self) -> None:
        """Shut down the LSP server."""
        try:
            await self._request("shutdown", {})
            await self._notify("exit", {})
        except Exception:
            pass
        if self._process.returncode is None:
            self._process.kill()
        if self._reader_task:
            self._reader_task.cancel()
