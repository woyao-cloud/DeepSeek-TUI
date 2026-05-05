"""Sidebar panel — plan, todo, agents, and tasks panels.

Port of `crates/tui/src/tui/sidebar.rs`.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Label


class SidebarPanel(VerticalScroll):
    """Sidebar with Plan, Todo, Agents, and Tasks panels."""

    DEFAULT_CSS = """
    SidebarPanel {
        width: 32;
        dock: left;
        background: $surface;
        border: solid $primary;
        overflow: auto;
    }

    .panel-title {
        text-style: bold;
        color: $primary;
        padding: 1 0 0 1;
    }

    .panel-content {
        margin: 0 1;
        padding: 0 0 1 0;
        border-bottom: solid $border;
    }

    .item {
        margin: 0 0 0 1;
        color: $text;
    }

    .item-pending {
        color: $text-muted;
    }

    .item-in-progress {
        color: $warning;
    }

    .item-completed {
        color: $success;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("📋 Plan", classes="panel-title")
        yield Static("No active plan.", id="plan-content", classes="panel-content")

        yield Label("✅ Todo", classes="panel-title")
        yield Static("No todo items.", id="todo-content", classes="panel-content")

        yield Label("🤖 Agents", classes="panel-title")
        yield Static("No running agents.", id="agents-content", classes="panel-content")

        yield Label("📊 Tasks", classes="panel-title")
        yield Static("No pending tasks.", id="tasks-content", classes="panel-content")

    def update_plan(self, steps: list[dict]) -> None:
        content = self.query_one("#plan-content")
        if not steps:
            content.update("No active plan.")
            return
        lines = []
        for s in steps:
            status = s.get("status", "pending")
            marker = {"pending": "○", "in_progress": "◉", "completed": "●"}.get(status, "○")
            lines.append(f"{marker} {s.get('step', '')[:50]}")
        content.update("\n".join(lines))

    def update_todos(self, items: list[dict]) -> None:
        content = self.query_one("#todo-content")
        if not items:
            content.update("No todo items.")
            return
        lines = []
        for i, item in enumerate(items):
            status = item.get("status", "pending")
            marker = {"pending": "○", "in_progress": "◉", "completed": "●"}.get(status, "○")
            lines.append(f"{marker} {item.get('content', '')[:40]}")
        content.update("\n".join(lines))

    def update_agents(self, agents: list[dict]) -> None:
        content = self.query_one("#agents-content")
        if not agents:
            content.update("No running agents.")
            return
        lines = [f"• {a.get('name', 'agent')} ({a.get('status', 'running')})" for a in agents]
        content.update("\n".join(lines))

    def update_tasks(self, tasks: list[dict]) -> None:
        content = self.query_one("#tasks-content")
        if not tasks:
            content.update("No pending tasks.")
            return
        lines = [f"• {t.get('name', 'task')}" for t in tasks]
        content.update("\n".join(lines))
