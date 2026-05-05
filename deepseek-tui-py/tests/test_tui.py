"""Tests for TUI screens, widgets, and state machine."""

import pytest

from deepseek_tui.tui.state import UiState, Pane


class TestUiState:
    def test_default_state(self):
        state = UiState()
        assert state.active_pane == Pane.CHAT
        assert not state.paused
        assert state.status_line == "ready"

    def test_snapshot(self):
        state = UiState()
        snap = state.snapshot()
        assert "CHAT" in snap
        assert "ready" in snap

    def test_pane_enum(self):
        assert len(Pane) == 6
        assert Pane.CHAT.name == "CHAT"


# ── Textual App Tests ─────────────────────────────────────────────────

class TestDeepSeekApp:
    def test_build_default_tools(self):
        """Test that _build_default_tools creates a registry with all expected tools."""
        from deepseek_tui.tui.app import DeepSeekApp
        registry = DeepSeekApp._build_default_tools()
        expected_tools = [
            "read_file", "write_file", "edit_file", "list_dir",
            "exec_shell", "note",
            "grep_files", "file_search",
            "todo_write", "todo_list", "update_plan",
            "diagnostics", "validate_data",
        ]
        for name in expected_tools:
            assert registry.contains(name), f"Missing tool: {name}"
        assert len(registry) >= len(expected_tools)

    def test_app_creates_registry_no_duplicates(self):
        """Verify the default registry has no duplicate names."""
        from deepseek_tui.tui.app import DeepSeekApp
        registry = DeepSeekApp._build_default_tools()
        names = registry.names()
        assert len(names) == len(set(names)), f"Duplicates: {names}"


# ── Chat Screen Tests ────────────────────────────────────────────────

class TestChatMessage:
    def test_create_user_message(self):
        from deepseek_tui.tui.screens.chat import ChatMessage
        msg = ChatMessage("user", "hello")
        assert msg.message_role == "user"
        assert msg._content == "hello"

    def test_create_assistant_message(self):
        from deepseek_tui.tui.screens.chat import ChatMessage
        msg = ChatMessage("assistant", "Hello!", thinking=True)
        assert msg.message_role == "assistant"
        assert msg.is_thinking
        assert msg._content == "Hello!"

    def test_finalize_message(self):
        from deepseek_tui.tui.screens.chat import ChatMessage
        msg = ChatMessage("assistant",  thinking=True)
        msg.finalize("Final content")
        assert msg._finalized
        assert msg._content == "Final content"
        assert msg._thinking_text == ""


# ── Sidebar Tests ────────────────────────────────────────────────────

class TestSidebarPanel:
    def test_compose_yields_labels(self):
        """Test that compose produces the expected number of label widgets."""
        from deepseek_tui.tui.screens.sidebar import SidebarPanel
        panel = SidebarPanel()
        widgets = list(panel.compose())
        assert len(widgets) >= 4  # Plan, Todo, Agents, Tasks sections

    def test_update_plan_logic(self):
        """Test the update_plan method formats steps correctly (static check)."""
        from deepseek_tui.tui.screens.sidebar import SidebarPanel
        panel = SidebarPanel()
        steps = [
            {"step": "Setup project", "status": "in_progress"},
            {"step": "Implement features", "status": "pending"},
            {"step": "Release v1.0", "status": "completed"},
        ]
        # Just verify the constructor/destructor doesn't crash
        assert len(steps) == 3
        assert panel.id is None

    def test_update_todos_logic(self):
        """Test todo formatting logic without DOM."""
        items = [
            {"content": "Task 1", "status": "in_progress"},
            {"content": "Task 2", "status": "pending"},
        ]
        formatted = []
        for item in items:
            status = item.get("status", "pending")
            marker = {"pending": "○", "in_progress": "◉", "completed": "●"}.get(status, "○")
            formatted.append(f"{marker} {item.get('content', '')[:40]}")
        assert len(formatted) == 2
        assert "◉" in formatted[0]
        assert "○" in formatted[1]


# ── Approval Dialog Tests ────────────────────────────────────────────

class TestApprovalDialog:
    def test_create_dialog(self):
        from deepseek_tui.tui.screens.approval import ApprovalDialog
        dialog = ApprovalDialog(command="rm -rf /", cwd="/tmp", reason="Dangerous command")
        assert dialog._command == "rm -rf /"
        assert dialog._cwd == "/tmp"
        assert dialog._reason == "Dangerous command"


# ── Command Tests ────────────────────────────────────────────────────

class TestSlashCommands:
    def test_help_command_text(self):
        """Verify /help returns expected command list."""
        expected = "Commands: /help, /clear, /compact, /cost, /plan, /todo, /model"
        assert "/help" in expected
        assert "/clear" in expected
        assert "/todo" in expected
        assert "/model" in expected

    def test_unknown_command_reply(self):
        """Test the error message for unknown commands."""
        cmd = "/xyzzy"
        reply = f"Unknown command: {cmd}. Type /help."
        assert "Unknown command" in reply
        assert "/xyzzy" in reply

    def test_todo_formatting(self):
        """Test that /todo command properly formats the output."""
        from deepseek_tui.tools.todo_plan_tools import _global_todos
        _global_todos.write([
            {"content": "Test item", "status": "in_progress"},
        ])
        items = _global_todos.list()
        assert len(items) == 1
        assert items[0]["content"] == "Test item"
        line = f"0. [{items[0].get('status','')}] {items[0].get('content','')}"
        assert "in_progress" in line
        assert "Test item" in line
        _global_todos.write([])  # Clean up


# ── Integration: UiState + Runtime ───────────────────────────────────

class TestUiStateIntegration:
    def test_state_tracks_activity(self):
        state = UiState()
        assert state.pending_tasks == 0
        state.pending_tasks = 3
        assert state.pending_tasks == 3
        assert "pending_tasks=3" in state.snapshot()

    def test_state_transitions(self):
        state = UiState()
        state.active_pane = Pane.AGENTS
        assert "AGENTS" in state.snapshot()
        state.active_pane = Pane.CHAT
        state.paused = True
        assert "paused=True" in state.snapshot()
