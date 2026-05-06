"""Tests for tool infrastructure, tools, execpolicy, hooks, MCP, and core runtime."""

import tempfile
from pathlib import Path

import pytest

from deepseek_tui.tools import (
    ToolSpec, ToolResult, ToolError, ToolRegistry, ToolRegistryBuilder,
    ToolContext, ToolParams,
    required_str, optional_str,
)


# ── Tool Base Tests ───────────────────────────────────────────────────

class TestToolError:
    def test_error_messages(self):
        assert "Failed to validate input" in str(ToolError.invalid_input("bad"))
        assert "missing required field 'path'" in str(ToolError.missing_field("path"))
        assert "escapes workspace" in str(ToolError.path_escape("/etc"))
        assert "timed out" in str(ToolError.timeout(30))

    def test_helpers(self):
        data = {"name": "test", "count": 5, "flag": True}
        assert required_str(data, "name") == "test"
        assert optional_str(data, "missing") is None
        assert not optional_str(data, "optional_field")  # truthy None
        with pytest.raises(ToolError):
            required_str(data, "nonexistent")


class TestToolResult:
    def test_success(self):
        r = ToolResult.success("ok")
        assert r.success and r.content == "ok"

    def test_error(self):
        r = ToolResult.error("fail")
        assert not r.success and r.content == "fail"

    def test_json(self):
        r = ToolResult.json({"a": 1})
        assert r.success and '"a"' in r.content


# ── Dummy Tool for Registry Tests ────────────────────────────────────

class EchoTool(ToolSpec):
    name = "echo"
    description = "Echoes input back"
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    async def execute(self, input, context):
        return ToolResult.success(f"Echo: {input.get('message', '')}")


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        r = ToolRegistry()
        r.register(EchoTool())
        return r

    def test_register_and_get(self, registry):
        assert registry.contains("echo")
        assert registry.get("echo") is not None

    def test_execute(self, registry):
        import asyncio
        ctx = ToolContext.new(Path("."))
        result = asyncio.run(registry.execute_full("echo", {"message": "hello"}, ctx))
        assert result.success
        assert "hello" in result.content

    def test_to_api_tools(self, registry):
        tools = registry.to_api_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

    def test_registry_builder(self):
        builder = ToolRegistryBuilder()
        builder.with_tool(EchoTool())
        registry = builder.build()
        assert registry.contains("echo")


class TestToolRegistryBuilderIntegration:
    """Test that all tool modules can be registered without conflict."""

    def test_file_tools(self):
        from deepseek_tui.tools.file_tools import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
        builder = ToolRegistryBuilder()
        builder.with_tool(ReadFileTool())
        builder.with_tool(WriteFileTool())
        builder.with_tool(EditFileTool())
        builder.with_tool(ListDirTool())
        r = builder.build()
        assert r.contains("read_file")
        assert r.contains("write_file")
        assert r.contains("edit_file")
        assert r.contains("list_dir")

    def test_shell_tools(self):
        from deepseek_tui.tools.shell_tools import ExecShellTool, ShellWaitTool, NoteTool
        builder = ToolRegistryBuilder()
        builder.with_tool(ExecShellTool())
        builder.with_tool(ShellWaitTool())
        builder.with_tool(NoteTool())
        r = builder.build()
        assert r.contains("exec_shell")
        assert r.contains("note")

    def test_search_git_tools(self):
        from deepseek_tui.tools.search_git_tools import (
            GrepFilesTool, FileSearchTool, GitStatusTool, GitDiffTool,
        )
        builder = ToolRegistryBuilder()
        builder.with_tool(GrepFilesTool())
        builder.with_tool(FileSearchTool())
        builder.with_tool(GitStatusTool())
        builder.with_tool(GitDiffTool())
        r = builder.build()
        assert r.contains("grep_files")
        assert r.contains("file_search")

    def test_todo_plan_tools(self):
        from deepseek_tui.tools.todo_plan_tools import (
            TodoWriteTool, TodoAddTool, TodoListTool, UpdatePlanTool,
        )
        builder = ToolRegistryBuilder()
        builder.with_tool(TodoWriteTool())
        builder.with_tool(TodoAddTool())
        builder.with_tool(TodoListTool())
        builder.with_tool(UpdatePlanTool())
        r = builder.build()
        assert r.contains("todo_write")

    def test_all_tools_no_name_conflicts(self):
        from deepseek_tui.tools.file_tools import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
        from deepseek_tui.tools.shell_tools import ExecShellTool, NoteTool
        from deepseek_tui.tools.search_git_tools import GrepFilesTool, FileSearchTool, GitStatusTool
        from deepseek_tui.tools.todo_plan_tools import TodoWriteTool, UpdatePlanTool
        from deepseek_tui.tools.web_validation_tools import ValidateDataTool

        builder = ToolRegistryBuilder()
        for tool in [ReadFileTool(), WriteFileTool(), EditFileTool(), ListDirTool(),
                      ExecShellTool(), NoteTool(),
                      GrepFilesTool(), FileSearchTool(), GitStatusTool(),
                      TodoWriteTool(), UpdatePlanTool(), ValidateDataTool()]:
            builder.with_tool(tool)
        r = builder.build()

        names = r.names()
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"
        assert len(names) >= 10


# ── File Tool Tests ──────────────────────────────────────────────────

class TestFileTools:
    @pytest.fixture
    def ctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield ToolContext.new(Path(tmp))

    async def test_write_and_read(self, ctx):
        from deepseek_tui.tools.file_tools import WriteFileTool, ReadFileTool
        w = WriteFileTool()
        r = ReadFileTool()

        result = await w.execute({"path": "test.txt", "content": "hello"}, ctx)
        assert result.success

        result = await r.execute({"path": "test.txt"}, ctx)
        assert result.success
        assert result.content == "hello"

    async def test_list_dir(self, ctx):
        from deepseek_tui.tools.file_tools import WriteFileTool, ListDirTool
        w = WriteFileTool()
        l = ListDirTool()

        await w.execute({"path": "a.txt", "content": "a"}, ctx)
        await w.execute({"path": "b.txt", "content": "b"}, ctx)

        result = await l.execute({"path": "."}, ctx)
        assert result.success
        assert "a.txt" in result.content
        assert "b.txt" in result.content

    async def test_path_escape_detected(self, ctx):
        from deepseek_tui.tools.file_tools import ReadFileTool
        r = ReadFileTool()
        result = await r.execute({"path": "../etc/passwd"}, ctx)
        assert not result.success
        assert "escape" in result.content.lower()


# ── ExecPolicy Tests ─────────────────────────────────────────────────

class TestExecPolicy:
    def test_allow_safe_command(self):
        from deepseek_tui.execpolicy import ExecPolicyEngine, ExecPolicyContext
        engine = ExecPolicyEngine()
        decision = engine.check(ExecPolicyContext(
            command="echo hello", cwd="/tmp", ask_for_approval=True
        ))
        assert decision.allow
        assert decision.requires_approval

    def test_deny_dangerous_command(self):
        from deepseek_tui.execpolicy import ExecPolicyEngine, ExecPolicyContext
        engine = ExecPolicyEngine()
        decision = engine.check(ExecPolicyContext(
            command="rm -rf /", cwd="/tmp"
        ))
        assert not decision.allow

    def test_allow_without_approval(self):
        from deepseek_tui.execpolicy import ExecPolicyEngine, ExecPolicyContext
        engine = ExecPolicyEngine()
        decision = engine.check(ExecPolicyContext(
            command="echo hi", cwd="/tmp", ask_for_approval=False
        ))
        assert decision.allow
        assert not decision.requires_approval


# ── Hooks Tests ──────────────────────────────────────────────────────

class TestHooks:
    def test_stdout_sink(self, capsys):
        from deepseek_tui.hooks import StdoutHookSink, HookEvent
        sink = StdoutHookSink()
        sink.emit(HookEvent("test", {"key": "value"}))
        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_dispatcher(self):
        from deepseek_tui.hooks import HookDispatcher
        dispatcher = HookDispatcher()
        # Should not raise
        import asyncio
        asyncio.run(dispatcher.emit("test", {"ok": True}))


# ── Core Runtime Tests ───────────────────────────────────────────────

class TestRuntime:
    def test_thread_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            from deepseek_tui.state import StateStore
            from deepseek_tui.core import ThreadManager
            store = StateStore(Path(tmp) / "state.db")
            mgr = ThreadManager(store)
            thread = mgr.spawn()
            assert thread.id is not None
            assert thread.preview == "New conversation"

            # Retrieve
            t = mgr.get(thread.id)
            assert t is not None
            assert t.id == thread.id

    def test_job_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            from deepseek_tui.state import StateStore
            from deepseek_tui.core import JobManager
            store = StateStore(Path(tmp) / "state.db")
            mgr = JobManager(store)
            job = mgr.enqueue("test-job")
            assert job.status == "queued"
            mgr.complete(job.id)
            assert mgr.list()[0].status == "completed"

    def test_runtime_create(self):
        from deepseek_tui.core import Runtime
        from deepseek_tui.config import ResolvedRuntimeOptions, ProviderKind
        from deepseek_tui.agent import ModelRegistry
        from deepseek_tui.state import StateStore
        from deepseek_tui.tools import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            config = ResolvedRuntimeOptions(
                provider=ProviderKind.DEEPSEEK,
                model="deepseek-v4-pro",
                api_key=None,
                base_url="https://api.deepseek.com",
                output_mode=None, log_level=None, telemetry=False,
                auth_mode=None, approval_policy=None, sandbox_mode=None,
            )
            registry = ModelRegistry()
            store = StateStore(Path(tmp) / "state.db")
            tools = ToolRegistry()

            rt = Runtime(config, registry, store, tools)
            assert rt.config.model == "deepseek-v4-pro"
            assert rt.llm_client is None  # No API key

    async def test_execute_tool(self):
        from deepseek_tui.core import Runtime
        from deepseek_tui.config import ResolvedRuntimeOptions, ProviderKind
        from deepseek_tui.agent import ModelRegistry
        from deepseek_tui.state import StateStore
        from deepseek_tui.tools import ToolRegistry
        from deepseek_tui.tools.file_tools import ReadFileTool, WriteFileTool

        with tempfile.TemporaryDirectory() as tmp:
            config = ResolvedRuntimeOptions(
                provider=ProviderKind.DEEPSEEK, model="deepseek-v4-pro",
                api_key=None, base_url="https://api.deepseek.com",
                output_mode=None, log_level=None, telemetry=False,
                auth_mode=None, approval_policy=None, sandbox_mode=None,
            )
            tools = ToolRegistry()
            tools.register(WriteFileTool())
            tools.register(ReadFileTool())
            store = StateStore(Path(tmp) / "state.db")

            rt = Runtime(config, ModelRegistry(), store, tools)

            # Write a file
            result = await rt.execute_tool("write_file", {
                "path": "test.txt", "content": "hello runtime"
            })
            assert "Written" in result

            # Read it back
            result = await rt.execute_tool("read_file", {"path": "test.txt"})
            assert "hello runtime" in result

    async def test_run_turn_executes_tool_and_continues(self):
        from deepseek_tui.agent import ModelRegistry
        from deepseek_tui.config import ProviderKind, ResolvedRuntimeOptions
        from deepseek_tui.core import Runtime
        from deepseek_tui.llm import ContentBlock, Delta, MessageDelta, MessageResponse, StreamEvent
        from deepseek_tui.state import StateStore
        from deepseek_tui.tools import ToolRegistry
        from deepseek_tui.tools.file_tools import ReadFileTool, WriteFileTool

        class FakeClient:
            provider_name = "fake"
            model = "deepseek-v4-pro"

            def __init__(self):
                self.requests = []

            async def create_message(self, request):
                raise AssertionError("run_turn should use streaming client path")

            async def create_message_stream(self, request):
                self.requests.append(request)
                saw_tool_result = any(
                    block.type == "tool_result"
                    for message in request.messages
                    for block in message.content
                )
                if not saw_tool_result:
                    yield StreamEvent(
                        type="content_block_start",
                        index=0,
                        content_block=ContentBlock(
                            type="tool_use",
                            id="call-1",
                            name="write_file",
                            input={},
                        ),
                    )
                    yield StreamEvent(
                        type="content_block_delta",
                        index=0,
                        delta=Delta(
                            type="input_json_delta",
                            partial_json='{"path":"turn.txt","content":"from tool loop"}',
                        ),
                    )
                    yield StreamEvent(type="content_block_stop", index=0)
                    yield StreamEvent(
                        type="message_delta",
                        message_delta=MessageDelta(stop_reason="tool_calls"),
                    )
                    yield StreamEvent(type="message_stop")
                    return

                yield StreamEvent(
                    type="content_block_start",
                    index=0,
                    content_block=ContentBlock(type="text", text=""),
                )
                yield StreamEvent(
                    type="content_block_delta",
                    index=0,
                    delta=Delta(type="text_delta", text="done"),
                )
                yield StreamEvent(type="content_block_stop", index=0)
                yield StreamEvent(
                    type="message_delta",
                    message_delta=MessageDelta(stop_reason="end_turn"),
                )
                yield StreamEvent(type="message_stop")

        with tempfile.TemporaryDirectory() as tmp:
            config = ResolvedRuntimeOptions(
                provider=ProviderKind.DEEPSEEK,
                model="deepseek-v4-pro",
                api_key=None,
                base_url="https://api.deepseek.com",
                output_mode=None,
                log_level=None,
                telemetry=False,
                auth_mode=None,
                approval_policy=None,
                sandbox_mode=None,
            )
            tools = ToolRegistry()
            tools.register(WriteFileTool())
            tools.register(ReadFileTool())
            rt = Runtime(config, ModelRegistry(), StateStore(Path(tmp) / "state.db"), tools)
            fake = FakeClient()
            rt.llm_client = fake
            thread = rt.thread_manager.spawn(cwd=Path(tmp))
            seen_events = []

            async def on_event(event):
                seen_events.append(event.event)

            response = await rt.run_turn(
                "write a file",
                thread=thread,
                event_callback=on_event,
            )

            assert response.output == "done"
            assert (Path(tmp) / "turn.txt").read_text() == "from tool loop"
            assert len(fake.requests) == 2
            assert "tool_call" in seen_events
            assert "response_delta" in seen_events
            assert seen_events[-1] == "response_end"
            assert any(event.event == "tool_result" for event in response.events)
            assert any(
                block.type == "thinking"
                for message in fake.requests[1].messages
                if message.role == "assistant"
                for block in message.content
            )
