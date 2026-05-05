"""Core runtime — port of `deepseek-core` crate.

Runtime, ThreadManager, JobManager, and TurnLoop orchestration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from deepseek_tui.agent import ModelRegistry, ModelResolution
from deepseek_tui.config import ProviderKind, ResolvedRuntimeOptions
from deepseek_tui.execpolicy import ExecPolicyEngine, ExecPolicyContext
from deepseek_tui.hooks import HookDispatcher
from deepseek_tui.llm import (
    LlmClient, DeepSeekClient, MessageRequest, MessageResponse,
    ContentBlock, Message, StreamEvent,
)
from deepseek_tui.mcp import McpManager
from deepseek_tui.protocol import (
    Thread, ThreadRequest, ThreadResponse, ThreadStatus,
    PromptRequest, PromptResponse, EventFrame,
    truncate_preview,
)
from deepseek_tui.state import (
    StateStore, ThreadMetadata, MessageRecord,
    PersistedThreadStatus,
    SessionSource, JobStateStatus, JobStateRecord,
    ThreadListFilters,
)
from deepseek_tui.tools import ToolRegistry, ToolContext, ToolParams


# ── Job Manager ───────────────────────────────────────────────────────

@dataclass
class JobRecord:
    id: str
    name: str
    status: str  # "queued" | "running" | "completed" | "failed" | "cancelled"
    progress: Optional[int] = None
    detail: Optional[str] = None
    retry_count: int = 0
    created_at: int = 0
    updated_at: int = 0


class JobManager:
    """Manages async job lifecycle."""

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._jobs: dict[str, JobRecord] = {}
        self._load_from_store()

    def _load_from_store(self) -> None:
        for rec in self._store.list_jobs(500):
            self._jobs[rec.id] = JobRecord(
                id=rec.id,
                name=rec.name,
                status=rec.status.value,
                progress=rec.progress,
                detail=rec.detail,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
            )

    def enqueue(self, name: str) -> JobRecord:
        now = int(datetime.now(timezone.utc).timestamp())
        job = JobRecord(
            id=f"job-{len(self._jobs)}-{now}",
            name=name,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self._jobs[job.id] = job
        self._persist(job)
        return job

    def set_running(self, job_id: str) -> None:
        if job := self._jobs.get(job_id):
            job.status = "running"
            job.updated_at = int(datetime.now(timezone.utc).timestamp())
            self._persist(job)

    def complete(self, job_id: str) -> None:
        if job := self._jobs.get(job_id):
            job.status = "completed"
            job.progress = 100
            job.updated_at = int(datetime.now(timezone.utc).timestamp())
            self._persist(job)

    def fail(self, job_id: str, detail: str = "") -> None:
        if job := self._jobs.get(job_id):
            job.status = "failed"
            job.detail = detail
            job.updated_at = int(datetime.now(timezone.utc).timestamp())
            self._persist(job)

    def list(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda j: -j.updated_at)

    def _persist(self, job: JobRecord) -> None:
        try:
            self._store.upsert_job(JobStateRecord(
                id=job.id,
                name=job.name,
                status=JobStateStatus(job.status),
                progress=job.progress,
                detail=job.detail,
                created_at=job.created_at,
                updated_at=job.updated_at,
            ))
        except Exception:
            pass


# ── Thread Manager ────────────────────────────────────────────────────

class ThreadManager:
    """Manages conversation threads."""

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._running: dict[str, Thread] = {}

    @property
    def store(self) -> StateStore:
        return self._store

    def spawn(self, model_provider: str = "deepseek", cwd: Optional[Path] = None,
              source: SessionSource = SessionSource.INTERACTIVE) -> Thread:
        now = int(datetime.now(timezone.utc).timestamp())
        thread = Thread(
            id=f"thread-{now}-{os.urandom(4).hex()}",
            preview="New conversation",
            model_provider=model_provider,
            created_at=now,
            updated_at=now,
            cwd=cwd or Path.cwd(),
            cli_version="0.1.0",
            source=source,
        )
        self._store.upsert_thread(ThreadMetadata(
            id=thread.id,
            preview=thread.preview,
            model_provider=thread.model_provider,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
            cwd=thread.cwd,
            cli_version=thread.cli_version,
            source=SessionSource(source.value),
        ))
        self._running[thread.id] = thread
        return thread

    def get(self, thread_id: str) -> Optional[Thread]:
        if thread_id in self._running:
            return self._running[thread_id]
        meta = self._store.get_thread(thread_id)
        if meta is None:
            return None
        return Thread(
            id=meta.id,
            preview=meta.preview,
            model_provider=meta.model_provider,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
            status=ThreadStatus(meta.status.value),
            cwd=meta.cwd,
            cli_version=meta.cli_version,
            source=meta.source,
            name=meta.name,
        )

    def list(self, include_archived: bool = False, limit: int = 50) -> list[Thread]:
        metas = self._store.list_threads(ThreadListFilters(
            include_archived=include_archived, limit=limit
        ))
        return [
            Thread(
                id=m.id, preview=m.preview, model_provider=m.model_provider,
                created_at=m.created_at, updated_at=m.updated_at,
                cwd=m.cwd, cli_version=m.cli_version, source=m.source, name=m.name,
            )
            for m in metas
        ]

    def touch(self, thread_id: str, input: str) -> None:
        now = int(datetime.now(timezone.utc).timestamp())
        preview = truncate_preview(input)
        self._store.append_message(thread_id, "user", input)
        self._store.upsert_thread(ThreadMetadata(
            id=thread_id, preview=preview, model_provider="",
            created_at=now, updated_at=now,
            cwd=Path.cwd(), cli_version="0.1.0",
        ))
        if thread_id in self._running:
            self._running[thread_id].updated_at = now
            self._running[thread_id].preview = preview


# ── Runtime ───────────────────────────────────────────────────────────

class Runtime:
    """Top-level runtime orchestrating threads, tools, LLM, and jobs."""

    def __init__(
        self,
        config: ResolvedRuntimeOptions,
        model_registry: ModelRegistry,
        state_store: StateStore,
        tool_registry: ToolRegistry,
        mcp_manager: Optional[McpManager] = None,
        exec_policy: Optional[ExecPolicyEngine] = None,
        hooks: Optional[HookDispatcher] = None,
    ) -> None:
        self.config = config
        self.model_registry = model_registry
        self.thread_manager = ThreadManager(state_store)
        self.tool_registry = tool_registry
        self.mcp_manager = mcp_manager or McpManager()
        self.exec_policy = exec_policy or ExecPolicyEngine()
        self.hooks = hooks or HookDispatcher()
        self.jobs = JobManager(state_store)

        # Build LLM client
        self.llm_client: Optional[LlmClient] = None
        if config.api_key:
            self.llm_client = DeepSeekClient(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
            )

    def _tool_context(self, thread: Optional[Thread] = None) -> ToolContext:
        return ToolContext(params=ToolParams(
            workspace=(thread.cwd if thread else Path.cwd()),
            cwd=(thread.cwd if thread else Path.cwd()),
            shell_allowed=True,
            network_allowed=True,
        ))

    async def handle_thread(self, req: ThreadRequest) -> ThreadResponse:
        kind = req.kind
        if kind == "create":
            thread = self.thread_manager.spawn()
            return ThreadResponse(thread_id=thread.id, status="created", thread=thread)
        elif kind == "start":
            thread = self.thread_manager.spawn(
                model_provider=req.model_provider or "deepseek",
                cwd=req.cwd,
            )
            return ThreadResponse(thread_id=thread.id, status="started", thread=thread)
        elif kind == "list":
            threads = self.thread_manager.list(req.include_archived, req.limit or 50)
            return ThreadResponse(thread_id="list", status="ok", threads=threads)
        elif kind == "read":
            thread = self.thread_manager.get(req.thread_id or "")
            return ThreadResponse(
                thread_id=req.thread_id or "",
                status="ok" if thread else "not_found",
                thread=thread,
            )
        elif kind == "message":
            tid = req.thread_id or ""
            input_text = req.input or ""
            self.thread_manager.touch(tid, input_text)
            return ThreadResponse(thread_id=tid, status="accepted")
        else:
            return ThreadResponse(thread_id=req.thread_id or "", status=f"unknown:{kind}")

    async def handle_prompt(self, req: PromptRequest) -> PromptResponse:
        """Handle a prompt request (non-streaming response)."""
        if self.llm_client is None:
            return PromptResponse(
                output="No API key configured",
                model="unknown",
                events=[EventFrame(event="error", message="No API key")],
            )

        # Resolve model
        resolution = self.model_registry.resolve(
            req.model or self.config.model,
            self.config.provider,
        )

        # Build messages
        messages = [
            Message(role="user", content=[ContentBlock(type="text", text=req.prompt)])
        ]

        # Attach tools
        tools = self.tool_registry.to_api_tools()

        llm_req = MessageRequest(
            model=resolution.resolved.id,
            messages=messages,
            tools=tools if tools else None,
            stream=True,
        )

        full_text = ""
        async for event in self.llm_client.create_message_stream(llm_req):
            if event.type == "content_block_delta" and event.delta:
                if event.delta.text:
                    full_text += event.delta.text

        return PromptResponse(
            output=full_text,
            model=resolution.resolved.id,
            events=[EventFrame(event="response_end")],
        )

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any],
        thread: Optional[Thread] = None,
    ) -> str:
        """Execute a tool by name with arguments."""
        ctx = self._tool_context(thread)
        try:
            result = await self.tool_registry.execute(tool_name, arguments, ctx)
            return result
        except Exception as e:
            return f"Error: {e}"
