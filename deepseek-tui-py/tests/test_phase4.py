"""Tests for Phase 4 — server, RLM, sub-agents, checkpoint/recovery."""

import json
import tempfile
from pathlib import Path

import pytest

from deepseek_tui.tools.rlm_tool import RlmTool
from deepseek_tui.tools.subagent import SubAgentManager, SubAgentRecord
from deepseek_tui.tools.checkpoint import CheckpointManager, OfflineQueue
from deepseek_tui.tools import ToolContext, ToolParams


class TestRlmTool:
    @pytest.fixture
    def ctx(self):
        return ToolContext(params=ToolParams(workspace=Path(".")))

    async def test_missing_fields(self, ctx):
        rlm = RlmTool()
        result = await rlm.execute({"task": "summarize"}, ctx)
        assert not result.success

        result = await rlm.execute({"content": "hello"}, ctx)
        assert not result.success

    async def test_default_processor_no_llm(self, ctx):
        rlm = RlmTool()
        result = await rlm.execute({
            "task": "summarize",
            "content": "A" * 5000,  # Long content
        }, ctx)
        assert result.success
        assert "Chunk" in result.content
        assert "Processed" in result.content

    async def test_custom_code_execution(self, ctx):
        rlm = RlmTool()
        result = await rlm.execute({
            "task": "count",
            "content": "hello world",
            "code": "print(f'Length: {len(PROMPT)}')",
        }, ctx)
        assert result.success
        assert "Length: 11" in result.content

    async def test_custom_code_error_handling(self, ctx):
        rlm = RlmTool()
        result = await rlm.execute({
            "task": "crash",
            "content": "test",
            "code": "raise ValueError('boom')",
        }, ctx)
        assert not result.success
        assert "boom" in result.content


class TestSubAgentManager:
    def test_create_and_list(self):
        mgr = SubAgentManager()
        assert mgr.list() == []

    def test_get_nonexistent(self):
        mgr = SubAgentManager()
        assert mgr.get("nonexistent") is None

    def test_record_dataclass(self):
        record = SubAgentRecord(
            id="agent-1",
            name="test-agent",
            status="completed",
            prompt="do something",
            result="done",
        )
        assert record.id == "agent-1"
        assert record.status == "completed"
        assert record.result == "done"


class TestCheckpointManager:
    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_save_and_load_checkpoint(self, tmp_dir):
        cpm = CheckpointManager(checkpoint_dir=tmp_dir)
        messages = [{"role": "user", "content": "Hello"}]
        path = cpm.save_checkpoint("thread-1", messages)
        assert path.exists()

        loaded = cpm.load_latest_checkpoint()
        assert loaded is not None
        assert loaded.thread_id == "thread-1"
        assert len(loaded.messages) == 1

    def test_clear_checkpoint(self, tmp_dir):
        cpm = CheckpointManager(checkpoint_dir=tmp_dir)
        cpm.save_checkpoint("thread-1", [])
        cpm.clear_checkpoint("thread-1")
        assert not (tmp_dir / "thread-1.json").exists()

    def test_empty_checkpoint_dir(self, tmp_dir):
        cpm = CheckpointManager(checkpoint_dir=tmp_dir)
        loaded = cpm.load_latest_checkpoint()
        assert loaded is None


class TestOfflineQueue:
    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Override the default path
            import deepseek_tui.tools.checkpoint as cp
            original = cp._CHECKPOINT_DIR
            cp._CHECKPOINT_DIR = Path(tmp)
            yield tmp
            cp._CHECKPOINT_DIR = original

    def test_enqueue_and_dequeue(self, tmp_dir):
        queue = OfflineQueue()
        assert queue.dequeue_all() == []

        queue.enqueue("hello", "thread-1")
        queue.enqueue("world")

        entries = queue.dequeue_all()
        assert len(entries) == 2
        assert entries[0]["prompt"] == "hello"
        assert entries[1]["prompt"] == "world"

        # After dequeue, it should be empty
        assert queue.dequeue_all() == []

    def test_peek_does_not_clear(self, tmp_dir):
        queue = OfflineQueue()
        queue.enqueue("test-prompt")

        entries = queue.peek()
        assert len(entries) == 1

        # Peek again — still there
        entries = queue.peek()
        assert len(entries) == 1


# ── Server Tests ─────────────────────────────────────────────────────

class TestServerApp:
    def test_create_app(self):
        """Test that the FastAPI app can be created."""
        from deepseek_tui.server.http_api import create_app
        from deepseek_tui.config import ResolvedRuntimeOptions, ProviderKind
        from deepseek_tui.agent import ModelRegistry
        from deepseek_tui.state import StateStore
        from deepseek_tui.tools import ToolRegistry
        from deepseek_tui.core import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            config = ResolvedRuntimeOptions(
                provider=ProviderKind.DEEPSEEK, model="deepseek-v4-pro",
                api_key=None, base_url="https://api.deepseek.com",
                output_mode=None, log_level=None, telemetry=False,
                auth_mode=None, approval_policy=None, sandbox_mode=None,
            )
            registry = ModelRegistry()
            store = StateStore(Path(tmp) / "state.db")
            tools = ToolRegistry()
            runtime = Runtime(config, registry, store, tools)

            app = create_app(runtime, config, store)

            # Check routes exist
            routes = [r.path for r in app.routes]
            assert "/healthz" in routes
            assert "/v1/threads" in routes
            assert "/v1/prompt" in routes
            assert "/v1/tools/call" in routes
            assert "/v1/status" in routes

    def test_healthz_route(self):
        from fastapi.testclient import TestClient
        from deepseek_tui.server.http_api import create_app
        from deepseek_tui.config import ResolvedRuntimeOptions, ProviderKind
        from deepseek_tui.agent import ModelRegistry
        from deepseek_tui.state import StateStore
        from deepseek_tui.tools import ToolRegistry
        from deepseek_tui.core import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            config = ResolvedRuntimeOptions(
                provider=ProviderKind.DEEPSEEK, model="deepseek-v4-pro",
                api_key=None, base_url="https://api.deepseek.com",
                output_mode=None, log_level=None, telemetry=False,
                auth_mode=None, approval_policy=None, sandbox_mode=None,
            )
            runtime = Runtime(config, ModelRegistry(), StateStore(Path(tmp) / "state.db"), ToolRegistry())
            app = create_app(runtime, config, StateStore(Path(tmp) / "state.db"))

            with TestClient(app) as client:
                response = client.get("/healthz")
                assert response.status_code == 200
                assert response.json()["status"] == "ok"


class TestStdioRpc:
    def test_rpc_server_creation(self):
        """Test that the StdioRpcServer can be instantiated."""
        from deepseek_tui.server.stdio_rpc import StdioRpcServer
        from deepseek_tui.config import ResolvedRuntimeOptions, ProviderKind
        from deepseek_tui.agent import ModelRegistry
        from deepseek_tui.state import StateStore
        from deepseek_tui.tools import ToolRegistry
        from deepseek_tui.core import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            config = ResolvedRuntimeOptions(
                provider=ProviderKind.DEEPSEEK, model="deepseek-v4-pro",
                api_key=None, base_url="https://api.deepseek.com",
                output_mode=None, log_level=None, telemetry=False,
                auth_mode=None, approval_policy=None, sandbox_mode=None,
            )
            runtime = Runtime(config, ModelRegistry(), StateStore(Path(tmp) / "state.db"), ToolRegistry())
            server = StdioRpcServer(runtime)
            assert server is not None
