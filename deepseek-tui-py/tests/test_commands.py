"""Tests for the slash command system."""

import pytest

from deepseek_tui.tui.commands import CommandRegistry, CommandInfo, CommandResult, dispatch, register, get_registry
from deepseek_tui.tui.commands.all_commands import register_all


class TestCommandRegistry:
    def setup_method(self):
        # Clear and re-register
        register_all()

    def test_register_and_get(self):
        registry = get_registry()
        cmd = registry.get("help")
        assert cmd is not None
        assert cmd.name == "help"

    def test_get_by_alias(self):
        registry = get_registry()
        cmd = registry.get("?")
        assert cmd is not None
        assert cmd.name == "help"

        cmd = registry.get("quit")
        assert cmd is not None
        assert cmd.name == "exit"

    def test_get_unknown_returns_none(self):
        registry = get_registry()
        cmd = registry.get("nonexistent")
        assert cmd is None  # fuzzy match only if close enough

    def test_fuzzy_match(self):
        """Typo like 'modle' should find 'model'."""
        registry = get_registry()
        cmd = registry.get("modle")
        assert cmd is not None
        assert cmd.name == "model"

    def test_all_returns_sorted(self):
        registry = get_registry()
        all_cmds = registry.all()
        names = [c.name for c in all_cmds]
        assert names == sorted(names)
        assert len(names) >= 40  # We should have 40+ commands

    def test_matching_prefix(self):
        registry = get_registry()
        cmds = registry.matching("co")
        names = [c.name for c in cmds]
        assert "cost" in names
        assert "config" in names
        assert "compact" in names
        assert "context" in names

    def test_no_duplicate_names_or_aliases(self):
        registry = get_registry()
        names = set()
        aliases = set()
        for cmd in registry.all():
            assert cmd.name not in names, f"Duplicate command name: {cmd.name}"
            names.add(cmd.name)
            for alias in cmd.aliases:
                assert alias not in names, f"Alias /{alias} collides with command name"
                assert alias not in aliases, f"Duplicate alias: {alias}"
                aliases.add(alias)


class TestCommandDispatch:
    def setup_method(self):
        register_all()

    def test_help_dispatch(self):
        result = dispatch("/help")
        assert not result.is_error
        assert "Available commands" in (result.message or "")

    def test_help_with_arg(self):
        result = dispatch("/help model")
        assert not result.is_error
        assert result.message and "model" in result.message

    def test_clear_dispatch(self):
        result = dispatch("/clear")
        assert not result.is_error
        assert result.action == "clear"

    def test_exit_dispatch(self):
        result = dispatch("/exit")
        assert result.action == "quit"

    def test_model_dispatch(self):
        result = dispatch("/model")
        assert not result.is_error
        assert result.message and "Current model" in result.message

    def test_cost_dispatch(self):
        result = dispatch("/cost")
        assert not result.is_error

    def test_unknown_command(self):
        result = dispatch("/zzzzz")
        assert result.is_error
        assert "Unknown command" in (result.message or "")

    def test_typo_suggestion(self):
        """Typo like 'modle' should give suggestion."""
        result = dispatch("/modle")
        assert result.message and "model" in result.message

    def test_plan_dispatch(self):
        result = dispatch("/plan")
        assert not result.is_error

    def test_settings_dispatch(self):
        result = dispatch("/settings")
        assert not result.is_error
        assert result.message and "settings" in result.message.lower() or "Current" in (result.message or "")

    def test_sessions_dispatch(self):
        result = dispatch("/sessions")
        assert not result.is_error

    def test_compact_dispatch(self):
        result = dispatch("/compact")
        assert not result.is_error
        assert "compacted" in (result.message or "")

    def test_hooks_dispatch(self):
        result = dispatch("/hooks")
        assert not result.is_error

    def test_subagents_dispatch(self):
        result = dispatch("/subagents")
        assert not result.is_error

    def test_links_dispatch(self):
        result = dispatch("/links")
        assert not result.is_error
        assert "platform.deepseek.com" in (result.message or "")

    def test_queue_dispatch(self):
        result = dispatch("/queue")
        assert not result.is_error

    def test_note_dispatch(self):
        result = dispatch("/note test")
        assert not result.is_error

    def test_tokens_dispatch(self):
        result = dispatch("/tokens")
        assert not result.is_error

    def test_system_dispatch(self):
        result = dispatch("/system")
        assert result.action == "show_system_prompt"

    def test_undo_dispatch(self):
        result = dispatch("/undo")
        assert not result.is_error

    def test_retry_dispatch(self):
        result = dispatch("/retry")
        assert not result.is_error

    def test_memory_dispatch(self):
        result = dispatch("/memory")
        assert not result.is_error

    def test_skills_dispatch(self):
        result = dispatch("/skills")
        assert not result.is_error

    def test_lsp_dispatch(self):
        result = dispatch("/lsp")
        assert not result.is_error

    def test_cache_dispatch(self):
        result = dispatch("/cache")
        assert not result.is_error

    def test_home_dispatch(self):
        result = dispatch("/home")
        assert not result.is_error

    def test_review_requires_arg(self):
        result = dispatch("/review")
        assert result.is_error

    def test_review_with_arg(self):
        result = dispatch("/review HEAD")
        assert not result.is_error

    def test_all_commands_dispatch_without_error(self):
        """Smoke test: every registered command dispatches without error."""
        registry = get_registry()
        for cmd in registry.all():
            if cmd.requires_argument():
                continue
            result = dispatch(f"/{cmd.name}")
            if result.is_error:
                # Only fail if the error is NOT about argument requirement
                assert "Usage:" in (result.message or "") or "type /help" in (result.message or ""), \
                    f"/{cmd.name} failed: {result.message}"
