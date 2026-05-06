"""Tests for the automation scheduling system."""

import tempfile
from pathlib import Path

import pytest

from deepseek_tui.tools.automation import (
    AutomationManager, AutomationRecord, AutomationRunRecord,
    AutomationStatus, RunStatus,
    _parse_rrule, _next_after,
    _now_iso,
)
from datetime import datetime, timezone, timedelta


class TestRRULE:
    def test_parse_hourly(self):
        r = _parse_rrule("FREQ=HOURLY;INTERVAL=2")
        assert r["freq"] == "HOURLY"
        assert r["interval_hours"] == 2

    def test_parse_hourly_with_byday(self):
        r = _parse_rrule("FREQ=HOURLY;INTERVAL=1;BYDAY=MO,TU")
        assert r["freq"] == "HOURLY"
        assert r["byday"] == [0, 1]

    def test_parse_weekly(self):
        r = _parse_rrule("FREQ=WEEKLY;BYDAY=MO,WE,FR;BYHOUR=9;BYMINUTE=30")
        assert r["freq"] == "WEEKLY"
        assert r["byday"] == [0, 2, 4]
        assert r["byhour"] == 9
        assert r["byminute"] == 30

    def test_parse_invalid_freq(self):
        with pytest.raises(ValueError, match="FREQ"):
            _parse_rrule("FREQ=DAILY")

    def test_parse_hourly_unknown_key(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _parse_rrule("FREQ=HOURLY;BYSECOND=5")

    def test_parse_weekly_missing_byday(self):
        with pytest.raises(ValueError, match="BYDAY"):
            _parse_rrule("FREQ=WEEKLY;BYHOUR=9")

    def test_parse_invalid_byday(self):
        with pytest.raises(ValueError, match="BYDAY"):
            _parse_rrule("FREQ=HOURLY;BYDAY=XX")

    def test_next_hourly(self):
        schedule = _parse_rrule("FREQ=HOURLY;INTERVAL=2")
        after = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        nxt = _next_after(schedule, after)
        assert nxt.hour == 12  # Next 2-hour aligned

    def test_next_weekly(self):
        schedule = _parse_rrule("FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0")
        # Saturday 2025-01-04
        after = datetime(2025, 1, 4, 10, 0, 0, tzinfo=timezone.utc)
        nxt = _next_after(schedule, after)
        assert nxt.weekday() == 0  # Monday
        # nxt.hour depends on local tz; just verify it's after `after`
        assert nxt > after


class TestAutomationManager:
    @pytest.fixture
    def manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield AutomationManager(Path(tmp))

    def test_create_and_get(self, manager):
        record = manager.create(
            name="Test Automation",
            prompt="Run this task",
            rrule="FREQ=HOURLY;INTERVAL=1",
        )
        assert record.name == "Test Automation"
        assert record.status == AutomationStatus.ACTIVE
        assert record.next_run_at is not None

        retrieved = manager.get(record.id)
        assert retrieved is not None
        assert retrieved.name == "Test Automation"

    def test_list(self, manager):
        manager.create(name="A1", prompt="p1", rrule="FREQ=HOURLY;INTERVAL=1")
        manager.create(name="A2", prompt="p2", rrule="FREQ=HOURLY;INTERVAL=2")
        all_a = manager.list()
        assert len(all_a) == 2

    def test_delete(self, manager):
        r = manager.create(name="Del", prompt="del", rrule="FREQ=HOURLY;INTERVAL=1")
        manager.delete(r.id)
        assert manager.get(r.id) is None

    def test_pause_resume(self, manager):
        r = manager.create(name="P", prompt="p", rrule="FREQ=HOURLY;INTERVAL=1")
        assert r.status == AutomationStatus.ACTIVE
        paused = manager.pause(r.id)
        assert paused.status == AutomationStatus.PAUSED
        assert paused.next_run_at is None

        resumed = manager.resume(r.id)
        assert resumed.status == AutomationStatus.ACTIVE
        assert resumed.next_run_at is not None

    def test_update_name(self, manager):
        r = manager.create(name="Old", prompt="p", rrule="FREQ=HOURLY;INTERVAL=1")
        updated = manager.update(r.id, name="New")
        assert updated.name == "New"

    def test_list_runs_empty(self, manager):
        r = manager.create(name="R", prompt="p", rrule="FREQ=HOURLY;INTERVAL=1")
        assert manager.list_runs(r.id) == []

    async def test_run_now(self, manager):
        r = manager.create(name="RN", prompt="p", rrule="FREQ=HOURLY;INTERVAL=1")
        run = await manager.run_now(r.id)
        assert run.status == RunStatus.RUNNING
        assert run.automation_id == r.id

    async def test_scheduler_tick(self, manager):
        r = manager.create(name="Sched", prompt="p", rrule="FREQ=HOURLY;INTERVAL=1")
        runs = await manager.scheduler_tick()
        assert isinstance(runs, list)

    def test_persist_across_instances(self, manager):
        r = manager.create(name="Persist", prompt="p", rrule="FREQ=HOURLY;INTERVAL=1")
        aid = r.id

        # New manager reading same directory
        manager2 = AutomationManager(manager._automations_dir.parent)
        retrieved = manager2.get(aid)
        assert retrieved is not None
        assert retrieved.name == "Persist"
