"""Tests for LSP integration and Skills system."""

import tempfile
from pathlib import Path

import pytest

from deepseek_tui.tools.lsp import (
    Diagnostic, Severity, DiagnosticBlock, render_blocks,
    Language, detect_language, server_for,
    LspConfig, LspManager,
)
from deepseek_tui.tools.skills import Skill, SkillRegistry, discover_in_workspace


# ── LSP Model Tests ──────────────────────────────────────────────────

class TestLspModels:
    def test_diagnostic_creation(self):
        d = Diagnostic(line=1, column=5, severity=Severity.ERROR, message="type error")
        assert d.line == 1
        assert d.severity == Severity.ERROR
        assert d.message == "type error"

    def test_diagnostic_block_render(self):
        block = DiagnosticBlock(
            file=Path("src/main.rs"),
            items=[
                Diagnostic(1, 5, Severity.ERROR, "expected type"),
                Diagnostic(3, 10, Severity.WARNING, "unused import"),
            ],
        )
        rendered = block.render()
        # Platform-agnostic: check for "main.rs" not full path
        assert "main.rs" in rendered
        assert "ERROR [1:5]" in rendered
        assert "WARN [3:10]" in rendered
        assert "expected type" in rendered
        assert "unused import" in rendered

    def test_diagnostic_block_truncate(self):
        items = [Diagnostic(i, 1, Severity.ERROR, f"err {i}") for i in range(10)]
        block = DiagnosticBlock(file=Path("test.rs"), items=items)
        block.truncate(3)
        assert len(block.items) == 3

    def test_render_blocks_concat(self):
        blocks = [
            DiagnosticBlock(file=Path("a.rs"), items=[
                Diagnostic(1, 1, Severity.ERROR, "err in a"),
            ]),
            DiagnosticBlock(file=Path("b.rs"), items=[
                Diagnostic(2, 2, Severity.ERROR, "err in b"),
            ]),
        ]
        rendered = render_blocks(blocks)
        assert "file=\"a.rs\"" in rendered
        assert "file=\"b.rs\"" in rendered

    def test_empty_block_renders_empty(self):
        block = DiagnosticBlock(file=Path("empty.rs"))
        assert block.render() == ""


# ── Language Registry Tests ──────────────────────────────────────────

class TestLanguageRegistry:
    def test_detect_rust(self):
        assert detect_language(Path("main.rs")) == Language.RUST

    def test_detect_python(self):
        assert detect_language(Path("app.py")) == Language.PYTHON

    def test_detect_typescript(self):
        assert detect_language(Path("component.tsx")) == Language.TYPESCRIPT
        assert detect_language(Path("lib.ts")) == Language.TYPESCRIPT

    def test_detect_javascript(self):
        assert detect_language(Path("index.js")) == Language.JAVASCRIPT

    def test_detect_go(self):
        assert detect_language(Path("server.go")) == Language.GO

    def test_detect_cpp(self):
        assert detect_language(Path("main.cpp")) == Language.CPP
        assert detect_language(Path("util.h")) == Language.CPP

    def test_detect_c(self):
        assert detect_language(Path("main.c")) == Language.C

    def test_detect_other(self):
        assert detect_language(Path("readme.md")) == Language.OTHER
        assert detect_language(Path("Makefile")) == Language.OTHER

    def test_server_for_rust(self):
        cmd, args = server_for(Language.RUST)
        assert cmd == "rust-analyzer"

    def test_server_for_python(self):
        cmd, args = server_for(Language.PYTHON)
        assert cmd == "pyright"

    def test_server_for_unknown(self):
        assert server_for(Language.OTHER) is None


# ── LspManager Tests ─────────────────────────────────────────────────

class TestLspManager:
    def test_disabled_manager(self):
        mgr = LspManager.disabled()
        assert not mgr.config().enabled

    def test_disabled_returns_none(self):
        mgr = LspManager(LspConfig(enabled=False), Path("/tmp"))
        import asyncio
        result = asyncio.run(mgr.diagnostics_for(Path("test.rs")))
        assert result is None

    def test_unknown_language_returns_none(self):
        mgr = LspManager(LspConfig(), Path("/tmp"))
        import asyncio
        result = asyncio.run(mgr.diagnostics_for(Path("readme.md")))
        assert result is None

    def test_lsp_config_resolve_command(self):
        config = LspConfig()
        cmd, args = config.resolve_command(Language.RUST)
        assert cmd == "rust-analyzer"

    def test_lsp_config_with_overrides(self):
        config = LspConfig(servers={"rust": ["custom-rls", "--lsp"]})
        cmd, args = config.resolve_command(Language.RUST)
        assert cmd == "custom-rls"
        assert args == ["--lsp"]


# ── Skills Tests ─────────────────────────────────────────────────────

class TestSkills:
    def test_parse_skill_with_frontmatter(self):
        content = (
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "allowed-tools: read_file, grep_files\n"
            "---\n"
            "Do something special\n"
        )
        registry = SkillRegistry()
        skill = registry._parse_skill(Path("/tmp/skills/test/SKILL.md"), content)
        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.body == "Do something special"
        assert len(skill.allowed_tools) == 2

    def test_parse_skill_missing_frontmatter(self):
        content = "Just text without frontmatter."
        registry = SkillRegistry()
        skill = registry._parse_skill(Path("test.md"), content)
        assert skill is None

    def test_parse_skill_missing_name(self):
        content = "---\ndescription: no name\n---\nbody"
        registry = SkillRegistry()
        skill = registry._parse_skill(Path("test.md"), content)
        assert skill is None

    def test_discover_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = SkillRegistry.discover(Path(tmp))
            assert registry.is_empty()

    def test_discover_nonexistent_directory(self):
        registry = SkillRegistry.discover(Path("/nonexistent/path"))
        assert registry.is_empty()

    def test_discover_single_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A skill\n---\nbody content"
            )
            registry = SkillRegistry.discover(Path(tmp))
            assert not registry.is_empty()
            assert len(registry.list()) == 1
            skill = registry.get("my-skill")
            assert skill is not None
            assert skill.description == "A skill"
            assert skill.body == "body content"

    def test_discover_in_workspace_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = discover_in_workspace(Path(tmp))
            assert registry.is_empty()

    def test_skill_registry_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_dir = Path(tmp) / "bad-skill"
            bad_dir.mkdir()
            (bad_dir / "SKILL.md").write_text("not valid")
            registry = SkillRegistry.discover(Path(tmp))
            assert registry.is_empty()
            assert len(registry.warnings) > 0
