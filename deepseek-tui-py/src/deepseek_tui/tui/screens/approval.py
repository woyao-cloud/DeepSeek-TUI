"""Approval dialog — tool execution confirmation UI.

Port of `crates/tui/src/tui/approval.rs`.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Label


class ApprovalDialog(ModalScreen[bool]):
    """Modal dialog for approving or denying tool execution."""

    class Approved(Message):
        def __init__(self, decision: str) -> None:
            self.decision = decision
            super().__init__()

    def __init__(
        self,
        command: str = "",
        cwd: str = "",
        reason: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._command = command
        self._cwd = cwd
        self._reason = reason

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("🔒 Tool Execution Approval Required", classes="panel-title")
            yield Static(f"Command: {self._command}", id="approval-command")
            yield Static(f"Directory: {self._cwd}", id="approval-cwd")
            if self._reason:
                yield Static(f"Reason: {self._reason}", id="approval-reason")
            yield Static("", id="approval-spacer")
            with Horizontal(classes="approval-buttons"):
                yield Button("✅ Approve", id="approve", variant="primary", classes="approval-button")
                yield Button("🔄 Approve Session", id="approve-session", variant="default", classes="approval-button")
                yield Button("❌ Deny", id="deny", variant="error", classes="approval-button")
                yield Button("⏹ Abort", id="abort", variant="warning", classes="approval-button")

    @on(Button.Pressed, "#approve")
    def approve(self) -> None:
        self.post_message(self.Approved("approved"))
        self.dismiss(True)

    @on(Button.Pressed, "#approve-session")
    def approve_session(self) -> None:
        self.post_message(self.Approved("approved_for_session"))
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def deny(self) -> None:
        self.post_message(self.Approved("denied"))
        self.dismiss(False)

    @on(Button.Pressed, "#abort")
    def abort(self) -> None:
        self.post_message(self.Approved("abort"))
        self.dismiss(False)
