"""SQLite-backed state store — port of `deepseek-state` StateStore.

Manages threads, messages, checkpoints, and jobs via SQLite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


# ── Enums ─────────────────────────────────────────────────────────────

class ThreadStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ARCHIVED = "archived"


class SessionSource(str, Enum):
    INTERACTIVE = "interactive"
    RESUME = "resume"
    FORK = "fork"
    API = "api"
    UNKNOWN = "unknown"


class JobStateStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Data Models ───────────────────────────────────────────────────────

@dataclass
class ThreadMetadata:
    id: str
    rollout_path: Optional[Path] = None
    preview: str = ""
    ephemeral: bool = False
    model_provider: str = "deepseek"
    created_at: int = 0
    updated_at: int = 0
    status: ThreadStatus = ThreadStatus.IDLE
    path: Optional[Path] = None
    cwd: Path = Path(".")
    cli_version: str = "0.1.0"
    source: SessionSource = SessionSource.UNKNOWN
    name: Optional[str] = None
    sandbox_policy: Optional[str] = None
    approval_mode: Optional[str] = None
    archived: bool = False
    archived_at: Optional[int] = None
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_origin_url: Optional[str] = None
    memory_mode: Optional[str] = None


@dataclass
class DynamicToolRecord:
    position: int
    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageRecord:
    id: int
    thread_id: str
    role: str
    content: str
    item: Optional[Any] = None
    created_at: int = 0


@dataclass
class CheckpointRecord:
    thread_id: str
    checkpoint_id: str
    state: Any = None
    created_at: int = 0


@dataclass
class JobStateRecord:
    id: str
    name: str
    status: JobStateStatus = JobStateStatus.QUEUED
    progress: Optional[int] = None
    detail: Optional[str] = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class ThreadListFilters:
    include_archived: bool = False
    limit: Optional[int] = None


# ── Session Index Entry ───────────────────────────────────────────────

@dataclass
class _SessionIndexEntry:
    thread_id: str
    thread_name: Optional[str] = None
    updated_at: int = 0
    rollout_path: Optional[Path] = None


# ── State Store ───────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT,
    preview TEXT NOT NULL,
    ephemeral INTEGER NOT NULL,
    model_provider TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    status TEXT NOT NULL,
    path TEXT,
    cwd TEXT NOT NULL,
    cli_version TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    sandbox_policy TEXT,
    approval_mode TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at INTEGER,
    git_sha TEXT,
    git_branch TEXT,
    git_origin_url TEXT,
    memory_mode TEXT
);
CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_archived_at ON threads(archived_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_archived_updated ON threads(archived, updated_at DESC);

CREATE TABLE IF NOT EXISTS thread_dynamic_tools (
    thread_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    input_schema TEXT NOT NULL,
    PRIMARY KEY (thread_id, position),
    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    item_json TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_thread_created_at ON messages(thread_id, created_at ASC);

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(thread_id, checkpoint_id),
    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_created_at ON checkpoints(thread_id, created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER,
    detail TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at DESC);
"""


def _default_state_db_path() -> Path:
    return Path.home() / ".deepseek" / "state.db"


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class StateStore:
    """SQLite-backed persistence for threads, messages, checkpoints, and jobs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _default_state_db_path()
        self._session_index_path = self._db_path.parent / "session_index.jsonl"
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── Connection ─────────────────────────────────────────────────

    def _conn(self):
        import sqlite3
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    # ── Thread CRUD ────────────────────────────────────────────────

    def upsert_thread(self, thread: ThreadMetadata) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO threads (
                    id, rollout_path, preview, ephemeral, model_provider,
                    created_at, updated_at, status, path, cwd,
                    cli_version, source, title, sandbox_policy, approval_mode,
                    archived, archived_at, git_sha, git_branch, git_origin_url, memory_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    rollout_path=excluded.rollout_path,
                    preview=excluded.preview,
                    ephemeral=excluded.ephemeral,
                    model_provider=excluded.model_provider,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    path=excluded.path,
                    cwd=excluded.cwd,
                    cli_version=excluded.cli_version,
                    source=excluded.source,
                    title=excluded.title,
                    sandbox_policy=excluded.sandbox_policy,
                    approval_mode=excluded.approval_mode,
                    archived=excluded.archived,
                    archived_at=excluded.archived_at,
                    git_sha=excluded.git_sha,
                    git_branch=excluded.git_branch,
                    git_origin_url=excluded.git_origin_url,
                    memory_mode=excluded.memory_mode""",
                (
                    thread.id,
                    str(thread.rollout_path) if thread.rollout_path else None,
                    thread.preview,
                    1 if thread.ephemeral else 0,
                    thread.model_provider,
                    thread.created_at,
                    thread.updated_at,
                    thread.status.value,
                    str(thread.path) if thread.path else None,
                    str(thread.cwd),
                    thread.cli_version,
                    thread.source.value,
                    thread.name,
                    thread.sandbox_policy,
                    thread.approval_mode,
                    1 if thread.archived else 0,
                    thread.archived_at,
                    thread.git_sha,
                    thread.git_branch,
                    thread.git_origin_url,
                    thread.memory_mode,
                ),
            )
            conn.commit()
            self._append_session_index(thread)
        finally:
            conn.close()

    def get_thread(self, thread_id: str) -> Optional[ThreadMetadata]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_thread(row)
        finally:
            conn.close()

    def list_threads(self, filters: Optional[ThreadListFilters] = None) -> list[ThreadMetadata]:
        f = filters or ThreadListFilters()
        conn = self._conn()
        try:
            if f.include_archived:
                rows = conn.execute(
                    "SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?",
                    (f.limit or 50,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM threads WHERE archived = 0 ORDER BY updated_at DESC LIMIT ?",
                    (f.limit or 50,),
                ).fetchall()
            return [self._row_to_thread(r) for r in rows]
        finally:
            conn.close()

    def mark_archived(self, thread_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE threads SET archived = 1, archived_at = ?, status = ? WHERE id = ?",
                (_now_ts(), ThreadStatus.ARCHIVED.value, thread_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_unarchived(self, thread_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE threads SET archived = 0, archived_at = NULL WHERE id = ?",
                (thread_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_thread(self, thread_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            conn.commit()
        finally:
            conn.close()

    # ── Dynamic Tools ──────────────────────────────────────────────

    def persist_dynamic_tools(self, thread_id: str, tools: list[DynamicToolRecord]) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM thread_dynamic_tools WHERE thread_id = ?", (thread_id,))
            for tool in tools:
                conn.execute(
                    "INSERT INTO thread_dynamic_tools(thread_id, position, name, description, input_schema) VALUES (?, ?, ?, ?, ?)",
                    (thread_id, tool.position, tool.name, tool.description, json.dumps(tool.input_schema)),
                )
            conn.commit()
        finally:
            conn.close()

    # ── Messages ───────────────────────────────────────────────────

    def append_message(
        self, thread_id: str, role: str, content: str, item: Optional[Any] = None
    ) -> int:
        conn = self._conn()
        try:
            now = _now_ts()
            item_json = json.dumps(item) if item is not None else None
            cursor = conn.execute(
                "INSERT INTO messages(thread_id, role, content, item_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, role, content, item_json, now),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def list_messages(self, thread_id: str, limit: Optional[int] = None) -> list[MessageRecord]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, thread_id, role, content, item_json, created_at FROM messages WHERE thread_id = ? ORDER BY created_at ASC LIMIT ?",
                (thread_id, limit or 500),
            ).fetchall()
            result = []
            for r in rows:
                item = json.loads(r["item_json"]) if r["item_json"] else None
                result.append(MessageRecord(
                    id=r["id"],
                    thread_id=r["thread_id"],
                    role=r["role"],
                    content=r["content"],
                    item=item,
                    created_at=r["created_at"],
                ))
            return result
        finally:
            conn.close()

    def clear_messages(self, thread_id: str) -> int:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            conn.commit()
            return conn.total_changes
        finally:
            conn.close()

    # ── Checkpoints ────────────────────────────────────────────────

    def save_checkpoint(self, thread_id: str, checkpoint_id: str, state: Any) -> None:
        conn = self._conn()
        try:
            state_json = json.dumps(state) if not isinstance(state, str) else state
            conn.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_id, state_json, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(thread_id, checkpoint_id) DO UPDATE SET state_json = excluded.state_json, created_at = excluded.created_at",
                (thread_id, checkpoint_id, state_json, _now_ts()),
            )
            conn.commit()
        finally:
            conn.close()

    def load_checkpoint(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[CheckpointRecord]:
        conn = self._conn()
        try:
            if checkpoint_id:
                row = conn.execute(
                    "SELECT thread_id, checkpoint_id, state_json, created_at FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT thread_id, checkpoint_id, state_json, created_at FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
            if row is None:
                return None
            return CheckpointRecord(
                thread_id=row["thread_id"],
                checkpoint_id=row["checkpoint_id"],
                state=json.loads(row["state_json"]),
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    def delete_checkpoint(self, thread_id: str, checkpoint_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Jobs ───────────────────────────────────────────────────────

    def upsert_job(self, job: JobStateRecord) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO jobs(id, name, status, progress, detail, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, status=excluded.status, "
                "progress=excluded.progress, detail=excluded.detail, "
                "created_at=excluded.created_at, updated_at=excluded.updated_at",
                (job.id, job.name, job.status.value, job.progress, job.detail, job.created_at, job.updated_at),
            )
            conn.commit()
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[JobStateRecord]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            return JobStateRecord(
                id=row["id"],
                name=row["name"],
                status=JobStateStatus(row["status"]),
                progress=row["progress"],
                detail=row["detail"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        finally:
            conn.close()

    def list_jobs(self, limit: Optional[int] = None) -> list[JobStateRecord]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?",
                (limit or 100,),
            ).fetchall()
            return [
                JobStateRecord(
                    id=r["id"],
                    name=r["name"],
                    status=JobStateStatus(r["status"]),
                    progress=r["progress"],
                    detail=r["detail"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def delete_job(self, job_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()

    # ── Session Index ──────────────────────────────────────────────

    def _append_session_index(self, thread: ThreadMetadata) -> None:
        entry = _SessionIndexEntry(
            thread_id=thread.id,
            thread_name=thread.name,
            updated_at=thread.updated_at,
            rollout_path=thread.rollout_path,
        )
        line = json.dumps({
            "thread_id": entry.thread_id,
            "thread_name": entry.thread_name,
            "updated_at": entry.updated_at,
            "rollout_path": str(entry.rollout_path) if entry.rollout_path else None,
        })
        self._session_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._session_index_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def find_thread_name_by_id(self, thread_id: str) -> Optional[str]:
        return self._build_session_index().get(thread_id)

    # ── Helpers ────────────────────────────────────────────────────

    def _row_to_thread(self, row) -> ThreadMetadata:
        return ThreadMetadata(
            id=row["id"],
            rollout_path=Path(row["rollout_path"]) if row["rollout_path"] else None,
            preview=row["preview"],
            ephemeral=bool(row["ephemeral"]),
            model_provider=row["model_provider"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=ThreadStatus(row["status"]),
            path=Path(row["path"]) if row["path"] else None,
            cwd=Path(row["cwd"]),
            cli_version=row["cli_version"],
            source=SessionSource(row["source"]),
            name=row["title"],
            sandbox_policy=row["sandbox_policy"],
            approval_mode=row["approval_mode"],
            archived=bool(row["archived"]),
            archived_at=row["archived_at"],
            git_sha=row["git_sha"],
            git_branch=row["git_branch"],
            git_origin_url=row["git_origin_url"],
            memory_mode=row["memory_mode"],
        )

    def _build_session_index(self) -> dict[str, Optional[str]]:
        if not self._session_index_path.exists():
            return {}
        result: dict[str, Optional[str]] = {}
        with open(self._session_index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                result[entry["thread_id"]] = entry.get("thread_name")
        return result
