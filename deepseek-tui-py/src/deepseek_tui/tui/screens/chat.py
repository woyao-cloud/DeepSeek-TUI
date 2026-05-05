"""Chat screen — main conversation view.

Port of `crates/tui/src/tui/transcript.rs`, `streaming/`, `user_input.rs`.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static, TextArea, Label
from rich.text import Text
from rich.markdown import Markdown
from typing import Optional


class ChatMessage(Static):
    """A single chat message bubble."""

    def __init__(self, role: str, content: str = "", thinking: bool = False, **kwargs):
        self.message_role = role
        self.is_thinking = thinking
        super().__init__(**kwargs)
        self._content = content
        self._thinking_text = ""
        self.tool_calls: list[str] = []
        self._finalized = False

    def on_mount(self) -> None:
        self._render()

    def _render(self) -> None:
        classes = "message-block "
        classes += "message-user" if self.message_role == "user" else "message-assistant"

        renderables = []

        # Thinking block
        if self._thinking_text and not self._finalized:
            thinking_rich = Text(f"🧠 {self._thinking_text}", style="italic dim")
            renderables.append(thinking_rich)

        # Tool calls
        for tc in self.tool_calls:
            renderables.append(Text(f"🔧 {tc}", style="bold green"))

        # Main content
        if self._content:
            if self.message_role == "user":
                renderables.append(Text(self._content))
            else:
                renderables.append(Markdown(self._content))

        if renderables:
            self.update(renderables[0] if len(renderables) == 1 else Text("\n\n").join(
                str(r) for r in renderables
            ))

    def update_content(self, content: str) -> None:
        self._content = content
        self._render()

    def update_thinking(self, thinking: str) -> None:
        self._thinking_text = thinking
        self._render()

    def add_tool_call(self, name: str) -> None:
        self.tool_calls.append(name)
        self._render()

    def finalize(self, content: str) -> None:
        self._finalized = True
        self._thinking_text = ""
        self._content = content
        classes = "message-block message-assistant"
        self.update(Markdown(content))
        self.classes = classes


class ChatContainer(Vertical):
    """Scrollable container for chat messages."""
    pass


class DeepSeekInput(TextArea):
    """Custom text area with slash command support."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    BINDINGS = [
        ("enter", "submit", "Submit"),
        ("escape", "cancel", "Cancel"),
    ]

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self.post_message(self.Submitted(text))
            self.text = ""

    def action_cancel(self) -> None:
        self.text = ""


class ChatScreen(Screen):
    """Main chat screen with message history and input."""

    BINDINGS = [
        ("ctrl+l", "focus_input", "Focus Input"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="main"):
            self.chat_container = ChatContainer(id="chat-container")
            yield self.chat_container
            with Container(id="input-container"):
                self.input_field = DeepSeekInput(
                    id="input-area",
                    placeholder="Type a message... (Ctrl+Enter to send)",
                    max_length=10000,
                    show_line_numbers=False,
                )
                yield self.input_field

    def on_mount(self) -> None:
        self.input_field.focus()

    @on(DeepSeekInput.Submitted)
    def handle_submit(self, event: DeepSeekInput.Submitted) -> None:
        text = event.text

        # Handle slash commands
        if text.startswith("/"):
            self.handle_command(text)
            return

        # Send to app for streaming
        app = self.app
        if hasattr(app, "stream_response"):
            app.stream_response(text)

    def handle_command(self, text: str) -> None:
        cmd = text.split()[0].lower()
        match cmd:
            case "/help":
                self._system_message("Commands: /help, /clear, /compact, /cost, /plan, /todo, /model")
            case "/clear":
                self.clear_messages()
                self._system_message("Conversation cleared.")
            case "/compact":
                self._system_message("Context compacted.")
            case "/cost":
                self._system_message("Cost tracking not yet implemented.")
            case "/plan":
                self._system_message("Current plan: nothing in progress.")
            case "/todo":
                from ...tools.todo_plan_tools import _global_todos
                items = _global_todos.list()
                if items:
                    lines = [f"{i}. [{item.get('status','')}] {item.get('content','')}" for i, item in enumerate(items)]
                    self._system_message("\n".join(lines))
                else:
                    self._system_message("No todo items.")
            case "/model":
                app = self.app
                model = getattr(app.runtime, "config", None)
                if model:
                    self._system_message(f"Model: {model.model}")
                else:
                    self._system_message("Model: not configured")
            case _:
                self._system_message(f"Unknown command: {cmd}. Type /help.")

    def _system_message(self, text: str) -> None:
        msg = ChatMessage("system", text)
        self.chat_container.mount(msg)
        msg.scroll_visible()

    def add_message(self, role: str, content: str = "", thinking: bool = False) -> ChatMessage:
        msg = ChatMessage(role, content, thinking)
        self.chat_container.mount(msg)
        msg.scroll_visible()
        return msg

    def update_message(self, msg: ChatMessage, content: str) -> None:
        msg.update_content(content)

    def update_thinking(self, msg: ChatMessage, thinking: str) -> None:
        msg.update_thinking(thinking)

    def finalize_message(self, msg: ChatMessage, content: str) -> None:
        msg.finalize(content)

    def clear_messages(self) -> None:
        self.chat_container.remove_children()

    def focus_input(self) -> None:
        self.input_field.focus()

    def action_focus_input(self) -> None:
        self.input_field.focus()
