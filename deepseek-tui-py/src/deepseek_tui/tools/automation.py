"""Automation scheduling — durable recurring tasks.

Port of `crates/tui/src/automation_manager.rs`.
Stores JSON records under `~/.deepseek/automations/`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ── Enums ─────────────────────────────────────────────────────────────

class AutomationStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"


class RunStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# ── RRULE Parsing ─────────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6,
}


def _parse_byday(value: str) -> list[int]:
    """Parse BYDAY=MO,WE,FR into list of weekday numbers (0=Mon)."""
    days = []
    for token in value.split(","):
        token = token.strip().upper()
        if token not in _WEEKDAY_MAP:
            raise ValueError(f"Invalid BYDAY value '{token}'")
        d = _WEEKDAY_MAP[token]
        if d not in days:
            days.append(d)
    return days


def _parse_rrule(rrule: str) -> dict[str, Any]:
    """Parse an RRULE string into a normalized dict."""
    parts: dict[str, str] = {}
    for raw in rrule.split(";"):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid RRULE segment '{item}'")
        k, v = item.split("=", 1)
        parts[k.strip().upper()] = v.strip().upper()

    freq = parts.get("FREQ")
    if freq not in ("HOURLY", "WEEKLY"):
        raise ValueError(f"Unsupported RRULE FREQ '{freq}'. Supported: HOURLY and WEEKLY")

    result: dict[str, Any] = {"freq": freq}

    if freq == "HOURLY":
        interval = int(parts.get("INTERVAL", "1"))
        if interval < 1:
            raise ValueError("INTERVAL must be >= 1")
        result["interval_hours"] = interval
        if "BYDAY" in parts:
            result["byday"] = _parse_byday(parts["BYDAY"])
        # Validate no unknown keys
        for k in parts:
            if k not in ("FREQ", "INTERVAL", "BYDAY"):
                raise ValueError(f"Unsupported RRULE field '{k}' for HOURLY")

    else:  # WEEKLY
        if "BYDAY" not in parts:
            raise ValueError("WEEKLY schedules require BYDAY")
        byday = _parse_byday(parts["BYDAY"])
        if not byday:
            raise ValueError("BYDAY cannot be empty")
        result["byday"] = byday
        byhour = int(parts.get("BYHOUR", "9"))
        byminute = int(parts.get("BYMINUTE", "0"))
        if not (0 <= byhour <= 23):
            raise ValueError("BYHOUR must be 0-23")
        if not (0 <= byminute <= 59):
            raise ValueError("BYMINUTE must be 0-59")
        result["byhour"] = byhour
        result["byminute"] = byminute
        for k in parts:
            if k not in ("FREQ", "BYDAY", "BYHOUR", "BYMINUTE"):
                raise ValueError(f"Unsupported RRULE field '{k}' for WEEKLY")

    return result


def _next_after(schedule: dict[str, Any], after: datetime) -> datetime:
    """Compute the next scheduled time after a given datetime."""
    freq = schedule["freq"]
    after_local = after.astimezone()

    if freq == "HOURLY":
        interval = schedule["interval_hours"]
        byday = schedule.get("byday")

        # Round up to next interval-aligned hour
        hours = after_local.hour
        aligned = (hours // interval + 1) * interval
        candidate = after_local.replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(hours=aligned - hours)

        if byday:
            for _ in range(24 * 21):
                if candidate.weekday() in byday:
                    return candidate.astimezone(timezone.utc)
                candidate += timedelta(hours=interval)
            raise ValueError("Could not find next run within BYDAY filter")
        return candidate.astimezone(timezone.utc)

    else:  # WEEKLY
        byday = schedule["byday"]
        byhour = schedule["byhour"]
        byminute = schedule["byminute"]

        for day_offset in range(1, 15):
            candidate_date = after_local.date() + timedelta(days=day_offset)
            if candidate_date.weekday() not in byday:
                continue
            candidate = datetime(
                candidate_date.year, candidate_date.month, candidate_date.day,
                byhour, byminute, 0, tzinfo=after_local.tzinfo,
            )
            if candidate > after_local:
                return candidate.astimezone(timezone.utc)
        raise ValueError("Could not compute next WEEKLY run")


# ── Data Models ───────────────────────────────────────────────────────

@dataclass
class AutomationRecord:
    id: str
    name: str
    prompt: str
    rrule: str
    status: AutomationStatus = AutomationStatus.ACTIVE
    cwds: list[str] = field(default_factory=list)
    created_at: str = ""  # ISO format
    updated_at: str = ""
    next_run_at: Optional[str] = None
    last_run_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "rrule": self.rrule,
            "status": self.status.value,
            "cwds": self.cwds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AutomationRecord:
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            prompt=d.get("prompt", ""),
            rrule=d.get("rrule", ""),
            status=AutomationStatus(d.get("status", "active")),
            cwds=d.get("cwds", []),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            next_run_at=d.get("next_run_at"),
            last_run_at=d.get("last_run_at"),
        )


@dataclass
class AutomationRunRecord:
    id: str
    automation_id: str
    scheduled_for: str
    status: RunStatus = RunStatus.QUEUED
    created_at: str = ""
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    task_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "automation_id": self.automation_id,
            "scheduled_for": self.scheduled_for,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "task_id": self.task_id,
            "error": self.error,
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AutomationRunRecord:
        return cls(
            id=d["id"],
            automation_id=d["automation_id"],
            scheduled_for=d.get("scheduled_for", ""),
            status=RunStatus(d.get("status", "queued")),
            created_at=d.get("created_at", ""),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            task_id=d.get("task_id"),
            error=d.get("error"),
        )


# ── AutomationManager ─────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_automations_dir() -> Path:
    return Path.home() / ".deepseek" / "automations"


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(content, encoding="utf-8")
    # Use replace() for cross-platform atomic write (rename() on Windows
    # raises FileExistsError when target exists).
    tmp.replace(path)


class AutomationManager:
    """Manages durable automation records and run history."""

    def __init__(self, root: Optional[Path] = None) -> None:
        root = root or _default_automations_dir()
        self._automations_dir = root / "automations"
        self._runs_dir = root / "runs"
        self._automations_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Automation CRUD ────────────────────────────────────────────

    def create(self, name: str, prompt: str, rrule: str,
               cwds: Optional[list[str]] = None,
               status: Optional[AutomationStatus] = None) -> AutomationRecord:
        if not name.strip():
            raise ValueError("Automation name is required")
        if not prompt.strip():
            raise ValueError("Automation prompt is required")

        import uuid
        schedule = _parse_rrule(rrule)  # Validate
        now = _now_iso()
        auto_status = status or AutomationStatus.ACTIVE

        next_run_at = None
        if auto_status == AutomationStatus.ACTIVE:
            next_run_at = _next_after(schedule, datetime.now(timezone.utc)).isoformat()

        record = AutomationRecord(
            id=str(uuid.uuid4()),
            name=name.strip(),
            prompt=prompt.strip(),
            rrule=rrule.strip().upper(),
            cwds=cwds or [],
            status=auto_status,
            created_at=now,
            updated_at=now,
            next_run_at=next_run_at,
        )
        self._save_automation(record)
        return record

    def get(self, automation_id: str) -> Optional[AutomationRecord]:
        path = self._automation_path(automation_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return AutomationRecord.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def list(self) -> list[AutomationRecord]:
        records = []
        for path in sorted(self._automations_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records.append(AutomationRecord.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    def update(self, automation_id: str, **kwargs) -> AutomationRecord:
        record = self.get(automation_id)
        if record is None:
            raise KeyError(f"Automation not found: {automation_id}")

        if "name" in kwargs:
            v = kwargs["name"]
            if not v.strip():
                raise ValueError("Name cannot be empty")
            record.name = v.strip()
        if "prompt" in kwargs:
            v = kwargs["prompt"]
            if not v.strip():
                raise ValueError("Prompt cannot be empty")
            record.prompt = v.strip()
        if "rrule" in kwargs:
            v = kwargs["rrule"].strip().upper()
            _parse_rrule(v)  # Validate
            record.rrule = v
            if record.status == AutomationStatus.ACTIVE:
                record.next_run_at = _next_after(
                    _parse_rrule(v), datetime.now(timezone.utc)
                ).isoformat()
        if "status" in kwargs:
            s = kwargs["status"]
            record.status = s
            if s == AutomationStatus.PAUSED:
                record.next_run_at = None
            else:
                schedule = _parse_rrule(record.rrule)
                record.next_run_at = _next_after(
                    schedule, datetime.now(timezone.utc)
                ).isoformat()
        record.updated_at = _now_iso()
        self._save_automation(record)
        return record

    def delete(self, automation_id: str) -> AutomationRecord:
        record = self.get(automation_id)
        if record is None:
            raise KeyError(f"Automation not found: {automation_id}")
        self._automation_path(automation_id).unlink(missing_ok=True)
        # Remove runs
        runs_dir = self._runs_dir_for(automation_id)
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
        return record

    def pause(self, automation_id: str) -> AutomationRecord:
        return self.update(automation_id, status=AutomationStatus.PAUSED)

    def resume(self, automation_id: str) -> AutomationRecord:
        return self.update(automation_id, status=AutomationStatus.ACTIVE)

    # ── Run Management ─────────────────────────────────────────────

    def list_runs(self, automation_id: str, limit: int = 20) -> list[AutomationRunRecord]:
        runs_dir = self._runs_dir_for(automation_id)
        if not runs_dir.exists():
            return []
        runs = []
        for path in sorted(runs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                runs.append(AutomationRunRecord.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    async def run_now(self, automation_id: str) -> AutomationRunRecord:
        """Execute an automation immediately (creates a run record)."""
        import uuid
        record = self.get(automation_id)
        if record is None:
            raise KeyError(f"Automation not found: {automation_id}")

        now = _now_iso()
        run = AutomationRunRecord(
            id=str(uuid.uuid4()),
            automation_id=automation_id,
            scheduled_for=now,
            status=RunStatus.RUNNING,
            created_at=now,
            started_at=now,
        )
        self._save_run(run)

        record.last_run_at = now
        record.updated_at = now
        if record.status == AutomationStatus.ACTIVE:
            schedule = _parse_rrule(record.rrule)
            record.next_run_at = _next_after(schedule, datetime.now(timezone.utc)).isoformat()
        self._save_automation(record)
        return run

    # ── Scheduler Tick ─────────────────────────────────────────────

    async def scheduler_tick(self) -> list[AutomationRunRecord]:
        """Check all active automations and enqueue due runs."""
        now = datetime.now(timezone.utc)
        new_runs: list[AutomationRunRecord] = []
        import uuid

        for record in self.list():
            if record.status != AutomationStatus.ACTIVE:
                continue
            if record.next_run_at is None:
                # Initialize next_run_at
                try:
                    schedule = _parse_rrule(record.rrule)
                    record.next_run_at = _next_after(schedule, now).isoformat()
                    self._save_automation(record)
                except Exception:
                    continue
                continue

            try:
                due_at = datetime.fromisoformat(record.next_run_at)
            except (ValueError, TypeError):
                continue

            if due_at > now:
                continue

            # Check idempotency
            existing = self.list_runs(record.id, limit=25)
            if any(r.scheduled_for == record.next_run_at for r in existing):
                # Already enqueued; advance next_run_at
                schedule = _parse_rrule(record.rrule)
                record.next_run_at = _next_after(schedule, due_at).isoformat()
                record.updated_at = _now_iso()
                self._save_automation(record)
                continue

            run = AutomationRunRecord(
                id=str(uuid.uuid4()),
                automation_id=record.id,
                scheduled_for=record.next_run_at,
                status=RunStatus.QUEUED,
                created_at=_now_iso(),
            )
            self._save_run(run)
            new_runs.append(run)

            # Advance next_run_at
            schedule = _parse_rrule(record.rrule)
            record.next_run_at = _next_after(schedule, due_at).isoformat()
            record.updated_at = _now_iso()
            self._save_automation(record)

        return new_runs

    # ── Persistence ────────────────────────────────────────────────

    def _automation_path(self, automation_id: str) -> Path:
        return self._automations_dir / f"{automation_id}.json"

    def _runs_dir_for(self, automation_id: str) -> Path:
        return self._runs_dir / automation_id

    def _run_path(self, automation_id: str, run_id: str) -> Path:
        return self._runs_dir_for(automation_id) / f"{run_id}.json"

    def _save_automation(self, record: AutomationRecord) -> None:
        _write_json_atomic(self._automation_path(record.id), record.to_dict())

    def _save_run(self, run: AutomationRunRecord) -> None:
        _write_json_atomic(self._run_path(run.automation_id, run.id), run.to_dict())
