"""DeepSeekApp — main Textual application.

Port of `crates/tui/src/tui/app.rs` with keybindings, screen routing, and app lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import RenderableType
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Header, Footer, Static

from ..core import Runtime
from ..tools import ToolRegistry
from ..tools.file_tools import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from ..tools.shell_tools import ExecShellTool, NoteTool
from ..tools.search_git_tools import GrepFilesTool, FileSearchTool
from ..tools.todo_plan_tools import (
    TodoWriteTool, TodoListTool, UpdatePlanTool, DiagnosticsTool,
)
from ..tools.web_validation_tools import ValidateDataTool

from .state import UiState, Pane


class DeepSeekApp(App):
    """Textual-based terminal UI for DeepSeek models."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 32;
        dock: left;
        background: $surface;
        border: solid $primary;
        overflow: auto;
    }

    #main {
        width: 1fr;
        height: 1fr;
    }

    #chat-container {
        height: 1fr;
        overflow: auto;
        border: solid $border;
        padding: 0 1;
    }

    #input-container {
        height: 3;
        dock: bottom;
        background: $surface;
        border-top: solid $primary;
    }

    #input-area {
        height: 3;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
        content-align: center middle;
    }

    .sidebar-panel {
        height: auto;
        margin: 0 1;
    }

    .sidebar-panel-title {
        text-style: bold;
        color: $primary;
        margin: 1 0 0 1;
    }

    .sidebar-item {
        margin: 0 1;
    }

    .message-block {
        margin: 0 1;
        padding: 1;
    }

    .message-user {
        background: $surface;
        border-left: solid $accent;
    }

    .message-assistant {
        background: $boost;
        border-left: solid $primary;
    }

    .thinking-block {
        color: $text-muted;
        background: $surface;
        border-left: solid $warning;
        margin: 0 1;
        padding: 0 1;
    }

    .tool-call {
        color: $secondary;
        background: $surface;
        border-left: solid $secondary;
        margin: 0 1;
        padding: 0 1;
    }

    #approval-dialog {
        width: 60;
        height: 16;
        border: thick $warning;
        background: $surface;
    }

    #approval-text {
        height: 8;
        overflow: auto;
    }

    .approval-button {
        width: 16;
    }

    Notification {
        width: 40;
        height: 3;
        background: $primary;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+p", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+l", "focus_input", "Focus Input"),
        Binding("escape", "close_dialog", "Close Dialog"),
        Binding("f1", "command_palette", "Commands"),
        Binding("ctrl+r", "resume_session", "Resume"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+d", "toggle_dark", "Toggle Dark"),
    ]

    def __init__(
        self,
        runtime: Optional[Runtime] = None,
        tool_registry: Optional[ToolRegistry] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ui_state = UiState()
        self.runtime = runtime
        self._tool_registry = tool_registry or self._build_default_tools()

    @staticmethod
    def _build_default_tools() -> ToolRegistry:
        """Build a default tool registry for the TUI session."""
        r = ToolRegistry()
        r.register(ReadFileTool())
        r.register(WriteFileTool())
        r.register(EditFileTool())
        r.register(ListDirTool())
        r.register(ExecShellTool())
        r.register(NoteTool())
        r.register(GrepFilesTool())
        r.register(FileSearchTool())
        r.register(TodoWriteTool())
        r.register(TodoListTool())
        r.register(UpdatePlanTool())
        r.register(DiagnosticsTool())
        r.register(ValidateDataTool())
        return r

    def compose(self) -> ComposeResult:
        from .screens.chat import ChatScreen
        from .screens.sidebar import SidebarPanel
        yield SidebarPanel()
        yield ChatScreen()

    def on_mount(self) -> None:
        self.title = "DeepSeek TUI"
        self.sub_title = "v0.1.0"
        self.notify("DeepSeek TUI ready. Type /help for commands.", timeout=3)

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_focus_input(self) -> None:
        from .screens.chat import ChatScreen
        screen = self.query_one(ChatScreen)
        screen.focus_input()

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_new_session(self) -> None:
        self.ui_state.messages = []
        self.ui_state.conversation_history = []
        self.ui_state.status_line = "new session"
        from .screens.chat import ChatScreen
        screen = self.query_one(ChatScreen)
        screen.clear_messages()

    def action_resume_session(self) -> None:
        self.notify("Resume: use --resume from CLI", timeout=3)

    def action_command_palette(self) -> None:
        self.notify(
            "Commands: /help, /clear, /compact, /cost, /plan, /todo", timeout=5
        )

    def action_close_dialog(self) -> None:
        from .screens.approval import ApprovalDialog
        dialog = self.query_one(ApprovalDialog)
        if dialog:
            dialog.dismiss(False)

    @work(exclusive=True)
    async def stream_response(self, prompt: str) -> None:
        """Stream a response from the LLM."""
        from .screens.chat import ChatScreen
        screen = self.query_one(ChatScreen)

        self.ui_state.status_line = "streaming..."
        screen.add_message("user", prompt)
        thinking_msg = screen.add_message("assistant", "", thinking=True)

        if self.runtime and self.runtime.llm_client:
            from ..llm import MessageRequest, Message
            from ..protocol import ContentBlock as ProtocolBlock

            messages = [
                Message(role="user", content=[ProtocolBlock(type="text", text=prompt)])
            ]
            tools = self._tool_registry.to_api_tools()

            req = MessageRequest(
                model=self.runtime.config.model,
                messages=messages,
                tools=tools if tools else None,
                stream=True,
            )

            text_accum = ""
            try:
                async for event in self.runtime.llm_client.create_message_stream(req):
                    if event.type == "content_block_delta" and event.delta:
                        if event.delta.text:
                            text_accum += event.delta.text
                            screen.update_message(thinking_msg, text_accum)
                        elif event.delta.thinking:
                            screen.update_thinking(thinking_msg, event.delta.thinking)
                    elif event.type == "content_block_stop":
                        pass
            except Exception as e:
                screen.update_message(thinking_msg, f"Error: {e}")
                self.ui_state.status_line = "error"

            if text_accum:
                screen.finalize_message(thinking_msg, text_accum)
                self.ui_state.status_line = "ready"
        else:
            screen.update_message(
                thinking_msg,
                "No API key configured. Use `deepseek login` or set DEEPSEEK_API_KEY.",
            )
            self.ui_state.status_line = "no API key"

    def update_status(self, text: str) -> None:
        self.ui_state.status_line = text
        status_bar = self.query_one("#status-bar")
        if status_bar:
            status_bar.update(Text(text, style="bold"))
