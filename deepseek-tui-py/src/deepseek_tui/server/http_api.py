"""HTTP/SSE API server — port of `deepseek-app-server` HTTP endpoints.

FastAPI-based server exposing thread, prompt, tool, job, and MCP endpoints.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..config import ConfigToml, ResolvedRuntimeOptions, ProviderKind
from ..agent import ModelRegistry
from ..state import StateStore
from ..tools import ToolRegistry
from ..core import Runtime


class ServerState:
    """Shared server state."""

    def __init__(
        self,
        runtime: Runtime,
        config: ResolvedRuntimeOptions,
        state_store: StateStore,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.state_store = state_store


def create_app(
    runtime: Runtime,
    config: ResolvedRuntimeOptions,
    state_store: StateStore,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Create a FastAPI application with all routes."""
    state = ServerState(runtime, config, state_store)
    app = FastAPI(title="DeepSeek TUI API", version="0.1.0")

    # CORS
    origins = cors_origins or [
        "http://localhost:3000",
        "http://localhost:1420",
        "tauri://localhost",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ─────────────────────────────────────────────────────

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "service": "deepseek-api"}

    # ── Thread endpoints ──────────────────────────────────────────

    class ThreadRequestPayload(BaseModel):
        kind: str
        thread_id: Optional[str] = None
        model_provider: Optional[str] = None
        cwd: Optional[str] = None
        input: Optional[str] = None
        include_archived: bool = False
        limit: Optional[int] = None

    @app.post("/v1/threads")
    async def handle_thread(payload: ThreadRequestPayload):
        from ..protocol import ThreadRequest as TR, ThreadResponse

        # Map to protocol request
        req_kwargs = {"kind": payload.kind}
        if payload.thread_id:
            req_kwargs["thread_id"] = payload.thread_id
        if payload.model_provider:
            req_kwargs["model_provider"] = payload.model_provider
        if payload.input:
            req_kwargs["input"] = payload.input
        req_kwargs["include_archived"] = payload.include_archived
        req_kwargs["limit"] = payload.limit

        req = TR(**req_kwargs)
        response = await state.runtime.handle_thread(req)
        return response.model_dump()

    @app.get("/v1/threads")
    async def list_threads(include_archived: bool = False, limit: int = 50):
        from ..protocol import ThreadRequest as TR

        req = TR(kind="list", include_archived=include_archived, limit=limit)
        response = await state.runtime.handle_thread(req)
        return response.model_dump()

    # ── Prompt endpoints ──────────────────────────────────────────

    class PromptPayload(BaseModel):
        prompt: str
        model: Optional[str] = None
        thread_id: Optional[str] = None
        stream: bool = False

    @app.post("/v1/prompt")
    async def handle_prompt(payload: PromptPayload):
        from ..protocol import PromptRequest as PR

        req = PR(prompt=payload.prompt, model=payload.model, thread_id=payload.thread_id)
        response = await state.runtime.handle_prompt(req)
        return response.model_dump()

    @app.post("/v1/prompt/stream")
    async def stream_prompt(request: Request, payload: PromptPayload):
        if not payload.stream:
            from ..protocol import PromptRequest as PR
            req = PR(prompt=payload.prompt, model=payload.model, thread_id=payload.thread_id)
            response = await state.runtime.handle_prompt(req)
            return response.model_dump()

        async def event_generator():
            if state.runtime.llm_client is None:
                yield {"event": "error", "data": json.dumps({"message": "No API key configured"})}
                return

            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            thread = (
                state.runtime.thread_manager.get(payload.thread_id)
                if payload.thread_id
                else None
            )

            async def on_event(event):
                await queue.put(("event", event))

            async def run_turn():
                try:
                    result = await state.runtime.run_turn(
                        payload.prompt,
                        thread=thread,
                        model=payload.model,
                        event_callback=on_event,
                    )
                    await queue.put(("result", result))
                except Exception as exc:
                    await queue.put(("error", exc))

            turn_task = asyncio.create_task(run_turn())

            try:
                while True:
                    kind, item = await queue.get()
                    if kind == "event":
                        if item.event == "response_delta" and item.delta:
                            yield {"event": "text", "data": json.dumps({"text": item.delta})}
                        elif item.event == "thinking_delta" and item.delta:
                            yield {
                                "event": "thinking",
                                "data": json.dumps({"thinking": item.delta}),
                            }
                        elif item.event == "tool_call":
                            yield {
                                "event": "tool_call",
                                "data": json.dumps(
                                    {
                                        "tool_name": item.tool_name,
                                        "arguments": item.arguments,
                                        "output": item.output,
                                    }
                                ),
                            }
                        elif item.event == "tool_result":
                            yield {
                                "event": "tool_result",
                                "data": json.dumps(
                                    {
                                        "tool_name": item.tool_name,
                                        "output": item.output,
                                    }
                                ),
                            }
                        elif item.event == "error":
                            yield {
                                "event": "error",
                                "data": json.dumps({"message": item.message or "runtime error"}),
                            }
                    elif kind == "result":
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {"output": item.output, "model": item.model}
                            ),
                        }
                        break
                    elif kind == "error":
                        yield {"event": "error", "data": json.dumps({"message": str(item)})}
                        break
            finally:
                await turn_task

        return EventSourceResponse(event_generator())

    # ── Tool endpoints ────────────────────────────────────────────

    class ToolCallPayload(BaseModel):
        name: str
        arguments: dict[str, Any] = {}
        thread_id: Optional[str] = None

    @app.post("/v1/tools/call")
    async def call_tool(payload: ToolCallPayload):
        thread = state.runtime.thread_manager.get(payload.thread_id) if payload.thread_id else None
        result = await state.runtime.execute_tool(payload.name, payload.arguments, thread)
        return {"ok": True, "result": result}

    # ── Job endpoints ─────────────────────────────────────────────

    @app.get("/v1/jobs")
    async def list_jobs():
        jobs = state.runtime.jobs.list()
        return {"ok": True, "jobs": [{"id": j.id, "name": j.name, "status": j.status} for j in jobs]}

    # ── MCP endpoints ─────────────────────────────────────────────

    @app.post("/v1/mcp/startup")
    async def mcp_startup():
        summary = await state.runtime.mcp_manager.start_all()
        return {"ok": True, "summary": summary}

    # ── Status endpoint ───────────────────────────────────────────

    @app.get("/v1/status")
    async def status():
        return {
            "ok": True,
            "model": state.config.model,
            "provider": state.config.provider.value,
            "has_api_key": state.config.api_key is not None,
            "tools": len(state.runtime.tool_registry),
        }

    return app


async def run_server(
    runtime: Runtime,
    config: ResolvedRuntimeOptions,
    state_store: StateStore,
    host: str = "127.0.0.1",
    port: int = 7878,
    cors_origins: Optional[list[str]] = None,
) -> None:
    """Run the HTTP/SSE API server."""
    import uvicorn

    app = create_app(runtime, config, state_store, cors_origins)
    config_uv = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config_uv)
    await server.serve()
