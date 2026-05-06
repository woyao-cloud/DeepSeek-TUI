"""Tests for the memory system and remember tool."""

import tempfile
from pathlib import Path

import pytest

from deepseek_tui.tools.memory import (
    load, as_system_block, compose_block, append_entry,
    MAX_MEMORY_SIZE,
)
from deepseek_tui.tools.remember_tool import RememberTool
from deepseek_tui.tools import ToolContext, ToolParams


class TestMemory:
    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "never-existed.md"
            assert load(path) is None

    def test_load_whitespace_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            path.write_text("   \n  \n")
            assert load(path) is None

    def test_load_real_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            path.write_text("remember the milk")
            result = load(path)
            assert result is not None
            assert "remember the milk" in result

    def test_as_system_block_wraps_content(self):
        block = as_system_block("note 1", Path("/tmp/m.md"))
        assert block is not None
        assert "<user_memory source=" in block
        assert "note 1" in block
        assert "</user_memory>" in block

    def test_as_system_block_returns_none_for_empty(self):
        assert as_system_block("   ", Path("/tmp/m.md")) is None

    def test_as_system_block_truncates_oversize(self):
        big = "x" * (MAX_MEMORY_SIZE + 100)
        block = as_system_block(big, Path("/tmp/m.md"))
        assert block is not None
        assert "(truncated" in block

    def test_compose_block_disabled(self):
        block = compose_block(False, Path("/tmp/m.md"))
        assert block is None

    def test_compose_block_missing_file(self):
        block = compose_block(True, Path("/tmp/does-not-exist.md"))
        assert block is None

    def test_append_entry_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            append_entry(path, "# remember the milk")
            assert path.exists()
            body = path.read_text()
            assert "remember the milk" in body
            assert body.startswith("- (")
            assert body.strip().endswith("remember the milk")

    def test_append_entry_appends_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            append_entry(path, "# first")
            append_entry(path, "second")
            body = path.read_text()
            assert "first" in body
            assert "second" in body
            assert body.count("- (") == 2

    def test_append_entry_rejects_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.md"
            with pytest.raises(ValueError, match="empty"):
                append_entry(path, "###")


class TestRememberTool:
    @pytest.fixture
    def ctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.md"
            tc = ToolContext(
                params=ToolParams(workspace=Path(tmp), cwd=Path(tmp)),
                extra={"memory_path": str(memory_path)},
            )
            yield tc

    async def test_remember_appends_entry(self, ctx):
        tool = RememberTool()
        result = await tool.execute({"note": "use 4 space indentation"}, ctx)
        assert result.success
        assert "4 space" in result.content

        memory_path = Path(ctx.extra["memory_path"])
        body = memory_path.read_text()
        assert "4 space" in body
        assert body.startswith("- (")

    async def test_remember_no_note(self, ctx):
        tool = RememberTool()
        result = await tool.execute({}, ctx)
        assert not result.success
        assert "Missing" in result.content

    async def test_remember_disabled(self):
        tc = ToolContext(params=ToolParams(workspace=Path(".")))
        tool = RememberTool()
        result = await tool.execute({"note": "test"}, tc)
        assert not result.success
        assert "disabled" in result.content

    async def test_remember_missing_note_field(self, ctx):
        tool = RememberTool()
        result = await tool.execute({}, ctx)
        assert not result.success
