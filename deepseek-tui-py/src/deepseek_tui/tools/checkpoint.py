"""Checkpoint and recovery — crash recovery + offline queue persistence.

Port of relevant parts from `crates/tui/src/compaction.rs` and `session_manager.rs`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_CHECKPOINT_DIR = Path.home() / ".deepseek" / "sessions" / "checkpoints"


@dataclass
class Checkpoint:
    """A serializable checkpoint snapshot."""
    thread_id: str
    messages: list[dict] = field(default_factory=list)
    tool_registry_state: Optional[dict] = None
    created_at: str = ""


class CheckpointManager:
    """Manages crash-recovery checkpoints and offline queues."""

    def __init__(self, checkpoint_dir: Optional[Path] = None) -> None:
        self._dir = checkpoint_dir or _CHECKPOINT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, thread_id: str, messages: list[dict]) -> Path:
        """Save a crash-recovery checkpoint."""
        checkpoint = Checkpoint(
            thread_id=thread_id,
            messages=messages,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        path = self._dir / f"{thread_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "thread_id": checkpoint.thread_id,
                "messages": checkpoint.messages,
                "created_at": checkpoint.created_at,
            }, f, indent=2, default=str)
        return path

    def load_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Load the most recent checkpoint."""
        checkpoints = sorted(self._dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not checkpoints:
            # Also check for the legacy latest.json
            legacy = self._dir / "latest.json"
            if legacy.exists():
                checkpoints = [legacy]
            else:
                return None

        try:
            with open(checkpoints[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint(
                thread_id=data.get("thread_id", ""),
                messages=data.get("messages", []),
                created_at=data.get("created_at", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def clear_checkpoint(self, thread_id: str) -> None:
        """Remove a checkpoint by thread ID."""
        path = self._dir / f"{thread_id}.json"
        if path.exists():
            path.unlink()


class OfflineQueue:
    """Queue for prompts submitted while offline/degraded."""

    def __init__(self) -> None:
        self._path = _CHECKPOINT_DIR / "offline_queue.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, prompt: str, thread_id: Optional[str] = None) -> None:
        """Add a prompt to the offline queue."""
        entry = {
            "prompt": prompt,
            "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def dequeue_all(self) -> list[dict]:
        """Read and clear all queued entries."""
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        # Clear the queue
        self._path.unlink(missing_ok=True)
        return entries

    def peek(self) -> list[dict]:
        """Read entries without clearing."""
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
