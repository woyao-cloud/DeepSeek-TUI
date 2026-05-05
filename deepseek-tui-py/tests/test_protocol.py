"""Tests for the protocol module."""

import json

from deepseek_tui.protocol import (
    Thread,
    ThreadStatus,
    SessionSource,
    ThreadRequest,
    ThreadResponse,
    EventFrame,
    ToolPayload,
    ToolOutput,
)


class TestThreadTypes:
    def test_thread_creation(self):
        t = Thread(
            id="thread-1",
            preview="Hello",
            model_provider="deepseek",
            created_at=1000,
            updated_at=1000,
            cwd="/tmp",
            cli_version="0.1.0",
        )
        assert t.id == "thread-1"
        assert t.status == ThreadStatus.RUNNING  # default
        assert t.source == SessionSource.UNKNOWN  # default

    def test_thread_serialization(self):
        t = Thread(
            id="thread-1",
            preview="Hello",
            model_provider="deepseek",
            created_at=1000,
            updated_at=1000,
            cwd="/tmp",
            cli_version="0.1.0",
        )
        data = t.model_dump()
        assert data["id"] == "thread-1"
        assert data["status"] == "running"

    def test_thread_status_enum(self):
        assert ThreadStatus.RUNNING.value == "running"
        assert ThreadStatus.ARCHIVED.value == "archived"


class TestEventFrame:
    def test_frame_creation(self):
        frame = EventFrame(event="response_start", response_id="resp-1")
        assert frame.event == "response_start"
        assert frame.response_id == "resp-1"

    def test_frame_serialization(self):
        frame = EventFrame(
            event="tool_call_start",
            response_id="resp-1",
            tool_name="read_file",
            arguments={"path": "test.txt"},
        )
        data = frame.model_dump()
        assert data["event"] == "tool_call_start"
        assert data["tool_name"] == "read_file"


class TestToolPayload:
    def test_payload_creation(self):
        payload = ToolPayload(
            type="function",
            arguments='{"path": "test.txt"}',
        )
        assert payload.type == "function"
        assert payload.arguments == '{"path": "test.txt"}'


class TestThreadRequest:
    def test_request_creation(self):
        req = ThreadRequest(kind="create", metadata={})
        assert req.kind == "create"

    def test_request_thread_id(self):
        req = ThreadRequest(kind="message", thread_id="thread-1", input="Hello")
        assert req.thread_id == "thread-1"
        assert req.input == "Hello"
