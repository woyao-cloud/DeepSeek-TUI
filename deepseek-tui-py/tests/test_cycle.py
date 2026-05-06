"""Tests for the Cycle Manager / Handoff system."""

import json
import tempfile
from pathlib import Path

import pytest

from deepseek_tui.core.cycle import (
    CycleConfig, should_advance_cycle,
    StructuredState, build_seed_messages,
    archive_cycle, open_archive, list_cycles,
    extract_carry_forward, enforce_briefing_cap,
    CycleBriefing, threshold_message,
    RecallArchiveTool,
)


class TestCycleConfig:
    def test_default_includes_v4_overrides(self):
        cfg = CycleConfig()
        assert cfg.enabled is True
        assert "deepseek-v4-pro" in cfg.per_model
        assert "deepseek-v4-flash" in cfg.per_model

    def test_threshold_falls_back_to_default(self):
        cfg = CycleConfig()
        assert cfg.threshold_for("deepseek-v4-pro") == 768_000
        assert cfg.threshold_for("unknown") == 768_000

    def test_threshold_uses_per_model_override(self):
        cfg = CycleConfig()
        cfg.per_model["deepseek-v4-pro"] = type(cfg.per_model["deepseek-v4-pro"])(
            threshold_tokens=80_000, briefing_max_tokens=2_000
        )
        assert cfg.threshold_for("deepseek-v4-pro") == 80_000
        assert cfg.briefing_max_for("deepseek-v4-pro") == 2_000


class TestShouldAdvance:
    def test_below_threshold_returns_false(self):
        cfg = CycleConfig()
        assert should_advance_cycle(50_000, 0, "deepseek-v4-pro", cfg, False) is False

    def test_at_threshold_returns_true(self):
        cfg = CycleConfig()
        assert should_advance_cycle(768_000, 0, "deepseek-v4-pro", cfg, False) is True

    def test_considers_headroom(self):
        cfg = CycleConfig()
        # Below 768K threshold but close to 1M window - 263K reserve
        assert should_advance_cycle(737_000, 263_168, "deepseek-v4-pro", cfg, False) is True

    def test_in_flight_blocks_advance(self):
        cfg = CycleConfig()
        assert should_advance_cycle(768_000 * 2, 0, "deepseek-v4-pro", cfg, True) is False

    def test_disabled_blocks_advance(self):
        cfg = CycleConfig(enabled=False)
        assert should_advance_cycle(768_000 * 2, 0, "deepseek-v4-pro", cfg, False) is False


class TestBriefing:
    def test_extract_carry_forward_with_tags(self):
        raw = (
            "Here is your handoff:\n"
            "<carry_forward>\n"
            "Decision A: chose X because Y.\n"
            "</carry_forward>\nDone."
        )
        assert extract_carry_forward(raw) == "Decision A: chose X because Y."

    def test_extract_carry_forward_missing_close(self):
        raw = "<carry_forward>\nDecision A: chose X."
        assert extract_carry_forward(raw) == "Decision A: chose X."

    def test_extract_carry_forward_no_tags(self):
        assert extract_carry_forward("  Decision A: chose X.  ") == "Decision A: chose X."

    def test_extract_carry_forward_case_insensitive(self):
        raw = "<CARRY_FORWARD>\nState here.\n</CARRY_FORWARD>"
        assert extract_carry_forward(raw) == "State here."

    def test_enforce_briefing_cap_truncates(self):
        big = "x" * 200
        bounded = enforce_briefing_cap(big, 10)
        assert bounded.startswith("x" * 40)
        assert "truncated" in bounded

    def test_enforce_briefing_cap_passes_short(self):
        assert enforce_briefing_cap("hello world", 100) == "hello world"


class TestStructuredState:
    def test_to_system_block_minimal(self):
        state = StructuredState(mode_label="agent", workspace="/tmp/ws")
        block = state.to_system_block()
        assert block is not None
        assert "Mode: `agent`" in block
        assert "Workspace: `/tmp/ws`" in block

    def test_to_system_block_with_plan(self):
        state = StructuredState(
            mode_label="agent",
            workspace="/tmp/ws",
            plan_steps=[{"step": "Setup", "status": "in_progress"}],
        )
        block = state.to_system_block()
        assert "[~]" in block
        assert "Setup" in block

    def test_to_system_block_with_todos(self):
        state = StructuredState(
            mode_label="agent",
            workspace="/tmp/ws",
            todo_items=[{"content": "Task 1", "status": "completed"}],
        )
        block = state.to_system_block()
        assert "[x]" in block
        assert "100% complete" in block


class TestSeedBuilder:
    def test_empty_when_none_provided(self):
        seeds = build_seed_messages(None, None, None)
        assert seeds == []

    def test_includes_state_briefing_and_pending(self):
        briefing = CycleBriefing(cycle=1, timestamp="2025-01-01", briefing_text="Decisions: A.", token_estimate=5)
        seeds = build_seed_messages(
            "## Cycle State\n- Mode: agent",
            briefing,
            "Continue working",
        )
        assert len(seeds) == 5  # state user + ack + briefing user + ack + pending user
        assert seeds[0]["role"] == "user"
        assert "[CYCLE STATE" in seeds[0]["content"][0]["text"]
        assert seeds[2]["role"] == "user"
        assert "[CYCLE BRIEFING" in seeds[2]["content"][0]["text"]
        assert seeds[4]["role"] == "user"
        assert seeds[4]["content"][0]["text"] == "Continue working"

    def test_skips_blank_pending(self):
        seeds = build_seed_messages("## State", None, "   ")
        assert len(seeds) == 2  # state + ack only


class TestArchive:
    def test_archive_and_open_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp)
            session_id = "test-session-123"

            messages = [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]
            path = archive_cycle(session_id, 1, messages, "deepseek-v4-pro", archive_dir)
            assert path.exists()
            assert path.name == "1.jsonl"

            header, loaded = open_archive(path)
            assert header["cycle"] == 1
            assert header["session_id"] == session_id
            assert header["message_count"] == 2
            assert len(loaded) == 2
            assert loaded[0]["role"] == "user"

    def test_open_archive_rejects_newer_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "99.jsonl"
            header = {
                "schema_version": 99,
                "cycle": 99,
                "session_id": "future",
                "model": "deepseek-v9",
                "started": "2025-01-01",
                "ended": "2025-01-01",
                "message_count": 0,
            }
            with open(path, "w") as f:
                f.write(json.dumps(header) + "\n")

            with pytest.raises(ValueError, match="newer than supported"):
                open_archive(path)

    def test_list_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp)
            session_id = "test"

            for i in range(3):
                archive_cycle(session_id, i + 1, [], "model", archive_dir)

            cycles = list_cycles(session_id, archive_dir)
            assert cycles == [1, 2, 3]

    def test_list_cycles_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert list_cycles("test", Path(tmp)) == []


class TestRecallArchiveTool:
    def test_missing_query(self):
        tool = RecallArchiveTool("sess-1")
        import asyncio
        result = asyncio.run(tool.execute({}))
        assert not result["ok"]

    def test_no_session_id(self):
        tool = RecallArchiveTool()
        import asyncio
        result = asyncio.run(tool.execute({"query": "test"}))
        assert not result["ok"]

    def test_search_with_results(self):
        # Test file-archiving directly rather than through the tool
        # (the tool uses ~/.deepseek/sessions/ which isn't accessible in test)
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp)
            messages = [
                {"role": "user", "content": [{"type": "text", "text": "Decision: use library A"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Confirmed, library A."}]},
            ]
            path = archive_cycle("sess-1", 1, messages, "deepseek-v4-pro", archive_dir)
            assert path.exists()

            # Verify the archive can be read back
            header, loaded = open_archive(path)
            assert header["cycle"] == 1
            assert len(loaded) == 2
            assert "library A" in loaded[0]["content"][0]["text"]

            # Tool search returns empty since it uses default dir
            tool = RecallArchiveTool("sess-1")
            import asyncio
            result = asyncio.run(tool.execute({"query": "library A", "max_cycles": 5}))
            assert result["ok"]
            # The tool searches ~/.deepseek/sessions/ not our tempdir


class TestThresholdMessage:
    def test_high_ratio_returns_message(self):
        msg = threshold_message(0.95)
        assert msg is not None
        assert "90%" in msg

    def test_medium_ratio_returns_lower_threshold(self):
        msg = threshold_message(0.85)
        assert msg is not None
        assert "80%" in msg

    def test_low_ratio_returns_none(self):
        msg = threshold_message(0.5)
        assert msg is None
