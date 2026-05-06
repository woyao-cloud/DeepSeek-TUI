"""Chat screen — main conversation view."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Input
from rich.text import Text
from rich.markdown import Markdown


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
        if self._thinking_text and not self._finalized:
            self.update(Text(f"🧠 {self._thinking_text}", style="italic dim"))
        elif self._content:
            if self.message_role == "user":
                self.update(Text(self._content))
            elif self._finalized:
                self.update(Markdown(self._content))
            else:
                self.update(Text(self._content[:200] + ("…" if len(self._content) > 200 else "")))
        else:
            self.update("")

    def update_content(self, content: str) -> None:
        self._content = content
        self._render()

    def update_thinking(self, thinking: str) -> None:
        self._thinking_text = thinking
        self._render()

    def finalize(self, content: str) -> None:
        self._finalized = True
        self._thinking_text = ""
        self._content = content
        self.update(Markdown(content))


class DeepSeekInput(Input):
    """Single-line input with Enter to submit."""

    def action_submit(self) -> None:
        """Override Input.action_submit to clear value after posting."""
        text = self.value.strip()
        if text:
            self.post_message(Input.Submitted(self.value))
            self.value = ""


class ChatScreen(Vertical):
    """Main chat view with message history and input."""

    DEFAULT_CSS = """
    ChatScreen {
        layout: vertical;
        height: 1fr;
        min-height: 5;
    }
    #chat-container {
        height: 1fr;
        overflow: auto;
        border: solid $border;
        padding: 0 1;
    }
    #input-area {
        dock: bottom;
        border-top: solid $primary;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        self.chat_container = Vertical(id="chat-container")
        yield self.chat_container
        self.input_field = DeepSeekInput(
            id="input-area",
            placeholder="Type a message and press Enter to send",
        )
        yield self.input_field

    def on_mount(self) -> None:
        self.input_field.focus()

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text.startswith("/"):
            self.handle_command(text)
            return
        app = self.app
        if hasattr(app, "stream_response"):
            app.stream_response(text)

    def handle_command(self, text: str) -> None:
        from ..commands import dispatch
        from ..commands.all_commands import register_all
        register_all()
        result = dispatch(text)
        if result.action:
            action = result.action
            if action == "clear":
                self.clear_messages()
                self._system_message("Conversation cleared.")
            elif action == "quit":
                self.app.exit()
            elif action == "logout":
                self._system_message("Logged out.")
            else:
                self._system_message(f"Action: {action}")
        elif result.message:
            self._system_message(result.message)
        elif result.is_error:
            self._system_message(result.message or "Unknown error")

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
