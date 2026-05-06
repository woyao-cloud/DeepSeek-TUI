"""Core runtime — port of `deepseek-core` crate.

Runtime, ThreadManager, JobManager, and TurnLoop orchestration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from deepseek_tui.agent import ModelRegistry, ModelResolution
from deepseek_tui.config import ProviderKind, ResolvedRuntimeOptions
from deepseek_tui.config.model import RunMode
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

@dataclass
class ToolUseState:
    id: str
    name: str
    input: Any = field(default_factory=dict)
    input_buffer: str = ""


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

        # Runtime mode (defaults to AGENT)
        self.mode = RunMode.AGENT

    def set_mode(self, mode: RunMode) -> None:
        """Switch runtime mode."""
        self.mode = mode

    def _tool_context(self, thread: Optional[Thread] = None) -> ToolContext:
        return ToolContext(params=ToolParams(
            workspace=(thread.cwd if thread else Path.cwd()),
            cwd=(thread.cwd if thread else Path.cwd()),
            shell_allowed=True,
            network_allowed=True,
        ))

    def _messages_for_thread(self, thread_id: str) -> list[Message]:
        messages: list[Message] = []
        for record in self.thread_manager.store.list_messages(thread_id):
            if isinstance(record.item, dict):
                try:
                    messages.append(Message.model_validate(record.item))
                    continue
                except Exception:
                    pass
            messages.append(
                Message(
                    role=record.role,
                    content=[ContentBlock(type="text", text=record.content)],
                )
            )
        return messages

    def _append_message(self, thread_id: str, message: Message) -> None:
        text = self._message_text(message)
        self.thread_manager.store.append_message(
            thread_id,
            message.role,
            text,
            message.model_dump(mode="json"),
        )

    @staticmethod
    def _message_text(message: Message) -> str:
        parts: list[str] = []
        for block in message.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
            elif block.type == "thinking" and block.thinking:
                parts.append(block.thinking)
            elif block.type == "tool_use":
                parts.append(f"[tool_use:{block.name}]")
            elif block.type == "tool_result" and block.content:
                parts.append(block.content)
        return "\n".join(parts)

    @staticmethod
    def _response_text(response: MessageResponse) -> str:
        return "\n".join(
            block.text or ""
            for block in response.content
            if block.type == "text" and block.text
        )

    @staticmethod
    def _tool_uses(response: MessageResponse) -> list[ContentBlock]:
        return [block for block in response.content if block.type == "tool_use"]

    async def _emit_event(
        self,
        event: EventFrame,
        events: list[EventFrame],
        event_callback: Optional[Callable[[EventFrame], Awaitable[None] | None]],
    ) -> None:
        events.append(event)
        if event_callback is None:
            return
        maybe_awaitable = event_callback(event)
        if maybe_awaitable is not None:
            await maybe_awaitable

    @staticmethod
    def _finalize_tool_state(tool_state: ToolUseState) -> dict[str, Any]:
        if tool_state.input_buffer.strip():
            try:
                parsed = json.loads(tool_state.input_buffer)
            except json.JSONDecodeError:
                parsed = tool_state.input
            if isinstance(parsed, dict):
                tool_state.input = parsed
        if isinstance(tool_state.input, dict):
            return tool_state.input
        return {}

    @staticmethod
    def _assistant_message_from_stream(
        text: str,
        thinking: str,
        tool_uses: list[ToolUseState],
    ) -> Message:
        blocks: list[ContentBlock] = []
        if thinking:
            blocks.append(ContentBlock(type="thinking", thinking=thinking))
        elif tool_uses:
            blocks.append(ContentBlock(type="thinking", thinking="(reasoning omitted)"))
        if text:
            blocks.append(ContentBlock(type="text", text=text))
        for tool_use in tool_uses:
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    id=tool_use.id,
                    name=tool_use.name,
                    input=tool_use.input,
                )
            )
        return Message(role="assistant", content=blocks)

    async def _stream_model_round(
        self,
        request: MessageRequest,
        events: list[EventFrame],
        event_callback: Optional[Callable[[EventFrame], Awaitable[None] | None]],
    ) -> tuple[Message, list[ToolUseState], str]:
        current_text = ""
        current_thinking = ""
        tool_uses: list[ToolUseState] = []
        tool_indices: dict[int, int] = {}

        async for event in self.llm_client.create_message_stream(request):
            if event.type == "content_block_start" and event.content_block:
                if event.content_block.type == "tool_use":
                    block_index = event.index if event.index is not None else 0
                    tool_indices[block_index] = len(tool_uses)
                    tool_uses.append(
                        ToolUseState(
                            id=event.content_block.id or f"call_{len(tool_uses)}",
                            name=event.content_block.name or "",
                            input=event.content_block.input
                            if isinstance(event.content_block.input, dict)
                            else {},
                        )
                    )
                continue

            if event.type == "content_block_delta" and event.delta:
                if event.delta.type == "text_delta" and event.delta.text:
                    current_text += event.delta.text
                    await self._emit_event(
                        EventFrame(event="response_delta", delta=event.delta.text),
                        events,
                        event_callback,
                    )
                elif event.delta.type == "thinking_delta" and event.delta.thinking:
                    current_thinking += event.delta.thinking
                    await self._emit_event(
                        EventFrame(event="thinking_delta", delta=event.delta.thinking),
                        events,
                        event_callback,
                    )
                elif event.delta.type == "input_json_delta" and event.delta.partial_json:
                    block_index = event.index if event.index is not None else -1
                    tool_index = tool_indices.get(block_index)
                    if tool_index is not None:
                        tool_uses[tool_index].input_buffer += event.delta.partial_json
                        self._finalize_tool_state(tool_uses[tool_index])
                continue

            if event.type == "content_block_stop":
                block_index = event.index if event.index is not None else -1
                tool_index = tool_indices.get(block_index)
                if tool_index is not None:
                    tool_state = tool_uses[tool_index]
                    arguments = self._finalize_tool_state(tool_state)
                    await self._emit_event(
                        EventFrame(
                            event="tool_call",
                            tool_name=tool_state.name,
                            arguments=arguments,
                            output={"tool_use_id": tool_state.id},
                        ),
                        events,
                        event_callback,
                    )
                continue

            if event.type == "message_stop":
                break

        assistant_message = self._assistant_message_from_stream(
            current_text,
            current_thinking,
            tool_uses,
        )
        return assistant_message, tool_uses, current_text

    async def run_turn(
        self,
        prompt: str,
        thread: Optional[Thread] = None,
        model: Optional[str] = None,
        max_tool_rounds: int = 8,
        event_callback: Optional[Callable[[EventFrame], Awaitable[None] | None]] = None,
    ) -> PromptResponse:
        """Run one user turn through the model/tool loop until final text."""
        if self.llm_client is None:
            return PromptResponse(
                output="No API key configured",
                model="unknown",
                events=[EventFrame(event="error", message="No API key")],
            )

        if thread is None:
            thread = self.thread_manager.spawn(
                model_provider=self.config.provider.value,
                cwd=Path.cwd(),
                source=SessionSource.API,
            )

        resolution = self.model_registry.resolve(model or self.config.model, self.config.provider)
        user_message = Message(
            role="user",
            content=[ContentBlock(type="text", text=prompt)],
        )
        self._append_message(thread.id, user_message)

        events: list[EventFrame] = []
        final_output = ""

        for _round in range(max_tool_rounds + 1):
            messages = self._messages_for_thread(thread.id)
            request = MessageRequest(
                model=resolution.resolved.id,
                messages=messages,
                tools=self.tool_registry.to_api_tools() or None,
                stream=True,
            )
            assistant_message, tool_uses, round_output = await self._stream_model_round(
                request,
                events,
                event_callback,
            )
            self._append_message(thread.id, assistant_message)

            if not tool_uses:
                final_output = round_output
                break

            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input if isinstance(tool_use.input, dict) else {}
                result = await self.execute_tool(tool_name, tool_input, thread)
                tool_result = Message(
                    role="user",
                    content=[
                        ContentBlock(
                            type="tool_result",
                            tool_use_id=tool_use.id,
                            content=result,
                            is_error=result.startswith("Error:")
                            or result.startswith("Blocked")
                            or result.startswith("Requires approval"),
                        )
                    ],
                )
                self._append_message(thread.id, tool_result)
                await self._emit_event(
                    EventFrame(
                        event="tool_result",
                        tool_name=tool_name,
                        output={"tool_use_id": tool_use.id, "result": result},
                    ),
                    events,
                    event_callback,
                )
        else:
            final_output = "Stopped: maximum tool rounds reached"
            await self._emit_event(
                EventFrame(event="error", message=final_output),
                events,
                event_callback,
            )

        await self._emit_event(
            EventFrame(event="response_end"),
            events,
            event_callback,
        )
        return PromptResponse(
            output=final_output,
            model=resolution.resolved.id,
            events=events,
        )

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
        thread = self.thread_manager.get(req.thread_id) if req.thread_id else None
        return await self.run_turn(req.prompt, thread=thread, model=req.model)

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any],
        thread: Optional[Thread] = None,
    ) -> str:
        """Execute a tool by name with arguments."""
        ctx = self._tool_context(thread)

        # Check execution policy with current mode
        policy_ctx = ExecPolicyContext(
            command=tool_name,
            cwd=str(ctx.params.cwd),
            tool=tool_name,
            ask_for_approval=bool(self.config.approval_policy),
        )
        decision = self.exec_policy.check(policy_ctx, self.mode)

        if not decision.allow:
            return f"Blocked by policy: {decision.reason or 'operation not allowed'}"

        if decision.requires_approval:
            # In AGENT mode, tools requiring approval are blocked until user approves
            # In PLAN mode, all tools are blocked
            if self.mode == RunMode.PLAN:
                return f"Blocked: PLAN mode does not allow tool execution"
            # In AGENT mode, we'd normally wait for user approval
            # For now, block tools that require approval
            return f"Requires approval: {decision.reason() or tool_name}"

        try:
            result = await self.tool_registry.execute(tool_name, arguments, ctx)
            return result
        except Exception as e:
            return f"Error: {e}"
