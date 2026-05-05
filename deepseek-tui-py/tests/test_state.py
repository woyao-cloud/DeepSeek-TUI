"""Tests for the state (SQLite persistence) module."""

import tempfile
from pathlib import Path

import pytest

from deepseek_tui.state import (
    StateStore,
    ThreadMetadata,
    MessageRecord,
    JobStateRecord,
    PersistedThreadStatus,
    SessionSource,
    JobStateStatus,
    ThreadListFilters,
)
ThreadStatus = PersistedThreadStatus


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "test_state.db"


class TestStateStore:
    def test_init_creates_db(self, db_path):
        store = StateStore(db_path)
        assert db_path.exists()

    def test_upsert_and_get_thread(self, db_path):
        store = StateStore(db_path)
        thread = ThreadMetadata(
            id="thread-1",
            preview="Hello",
            model_provider="deepseek",
            created_at=1000,
            updated_at=1000,
            cwd=Path("/tmp"),
            cli_version="0.1.0",
        )
        store.upsert_thread(thread)
        retrieved = store.get_thread("thread-1")
        assert retrieved is not None
        assert retrieved.id == "thread-1"
        assert retrieved.preview == "Hello"

    def test_list_threads(self, db_path):
        store = StateStore(db_path)
        for i in range(3):
            store.upsert_thread(ThreadMetadata(
                id=f"thread-{i}",
                preview=f"Thread {i}",
                model_provider="deepseek",
                created_at=1000 + i,
                updated_at=1000 + i,
                cwd=Path("/tmp"),
                cli_version="0.1.0",
            ))
        threads = store.list_threads(ThreadListFilters(limit=10))
        assert len(threads) == 3

    def test_archive_thread(self, db_path):
        store = StateStore(db_path)
        store.upsert_thread(ThreadMetadata(
            id="thread-1",
            preview="Hello",
            model_provider="deepseek",
            created_at=1000,
            updated_at=1000,
            cwd=Path("/tmp"),
            cli_version="0.1.0",
        ))
        store.mark_archived("thread-1")
        thread = store.get_thread("thread-1")
        assert thread is not None
        assert thread.archived
        assert thread.status == ThreadStatus.ARCHIVED

    def test_messages_crud(self, db_path):
        store = StateStore(db_path)
        store.upsert_thread(ThreadMetadata(
            id="thread-1",
            preview="Hello",
            model_provider="deepseek",
            created_at=1000,
            updated_at=1000,
            cwd=Path("/tmp"),
            cli_version="0.1.0",
        ))
        msg_id = store.append_message("thread-1", "user", "Hello!")
        assert msg_id > 0

        messages = store.list_messages("thread-1")
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello!"

    def test_checkpoints(self, db_path):
        store = StateStore(db_path)
        store.upsert_thread(ThreadMetadata(
            id="thread-1",
            preview="Hello",
            model_provider="deepseek",
            created_at=1000,
            updated_at=1000,
            cwd=Path("/tmp"),
            cli_version="0.1.0",
        ))
        store.save_checkpoint("thread-1", "cp-1", {"state": "ok"})

        cp = store.load_checkpoint("thread-1", "cp-1")
        assert cp is not None
        assert cp.checkpoint_id == "cp-1"
        assert cp.state["state"] == "ok"

    def test_jobs_crud(self, db_path):
        store = StateStore(db_path)
        job = JobStateRecord(
            id="job-1",
            name="test-job",
            status=JobStateStatus.QUEUED,
            created_at=1000,
            updated_at=1000,
        )
        store.upsert_job(job)

        retrieved = store.get_job("job-1")
        assert retrieved is not None
        assert retrieved.name == "test-job"
        assert retrieved.status == JobStateStatus.QUEUED

        jobs = store.list_jobs()
        assert len(jobs) == 1
