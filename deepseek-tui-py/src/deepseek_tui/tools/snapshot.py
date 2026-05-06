"""Workspace snapshot system — side-git repo for pre/post-turn rollback.

Port of `crates/tui/src/snapshot/` (paths.rs, repo.rs, prune.rs).

Snapshots live in `~/.deepseek/snapshots/<project_hash>/<worktree_hash>/.git`
and never touch the user's own `.git` directory.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Path Resolution ───────────────────────────────────────────────────

def _stable_hash(path: Path) -> str:
    """FNV-1a-like stable hash (deterministic, not cryptographic)."""
    h = 0xcbf29ce484222325
    for b in str(path).encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return f"{h:016x}"


def snapshot_dir_for(workspace: Path, home: Optional[Path] = None) -> Path:
    """Compute the snapshot directory for a workspace."""
    home = home or Path.home()
    canonical = workspace.resolve()
    project_hash = _stable_hash(canonical)
    worktree_hash = _stable_hash(canonical)
    return home / ".deepseek" / "snapshots" / project_hash / worktree_hash


def snapshot_git_dir(workspace: Path) -> Path:
    """Resolve the .git directory inside the snapshot dir."""
    return snapshot_dir_for(workspace) / ".git"


# ── Snapshot Types ────────────────────────────────────────────────────

@dataclass
class SnapshotId:
    sha: str

    def __str__(self) -> str:
        return self.sha


@dataclass
class Snapshot:
    id: SnapshotId
    label: str
    timestamp: int  # Unix seconds


# ── Error ─────────────────────────────────────────────────────────────

class SnapshotError(Exception):
    pass


# ── SnapshotRepo ──────────────────────────────────────────────────────

class SnapshotRepo:
    """Side-git repository wrapper for workspace snapshots."""

    def __init__(self, workspace: Path) -> None:
        self._work_tree = workspace.resolve()
        self._git_dir = snapshot_git_dir(self._work_tree)
        self._init_if_needed()

    @property
    def git_dir(self) -> Path:
        return self._git_dir

    @property
    def work_tree(self) -> Path:
        return self._work_tree

    # ── Init ───────────────────────────────────────────────────────

    def _init_if_needed(self) -> None:
        if self._git_dir.exists():
            return
        parent = self._git_dir.parent
        parent.mkdir(parents=True, exist_ok=True)

        self._git("init", "--quiet", str(parent))
        self._git("config", "user.name", "deepseek-snapshots")
        self._git("config", "user.email", "snapshots@deepseek-tui.local")
        self._git("config", "gc.auto", "0")
        self._git("config", "core.autocrlf", "false")

    # ── Git wrapper ────────────────────────────────────────────────

    def _git(self, *args: str) -> str:
        """Run a git command against the side repo. Returns stdout."""
        cmd = ["git", "--git-dir", str(self._git_dir),
               "--work-tree", str(self._work_tree)] + list(args)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise SnapshotError(
                    f"git {' '.join(args)} failed: {result.stderr.strip()}"
                )
            return result.stdout.strip()
        except FileNotFoundError:
            raise SnapshotError("git not found — is it installed?")
        except subprocess.TimeoutExpired:
            raise SnapshotError("git command timed out")

    # ── Core Operations ────────────────────────────────────────────

    def snapshot(self, label: str) -> SnapshotId:
        """Take a snapshot of the current workspace state."""
        # Stage all changes
        self._git("add", "-A")

        # Write tree
        tree = self._git("write-tree")

        # Get parent if exists
        parent = ""
        try:
            parent = self._git("rev-parse", "--verify", "HEAD")
        except SnapshotError:
            parent = ""

        # Commit tree
        args = ["commit-tree", tree]
        if parent:
            args += ["-p", parent]
        args += ["-m", label]
        sha = self._git(*args)

        # Update HEAD
        self._git("update-ref", "HEAD", sha)

        return SnapshotId(sha)

    def restore(self, sid: SnapshotId) -> None:
        """Restore workspace to a snapshot's state."""
        current_paths = self._tree_paths("HEAD")
        target_paths = self._tree_paths(sid.sha)

        # Checkout snapshot state
        self._git("checkout", sid.sha, "--", ":/")

        # Remove files that existed at HEAD but not in the target
        for rel in current_paths - target_paths:
            rel_str = str(rel)
            if ".." in rel_str or rel_str.startswith("/"):
                continue
            path = self._work_tree / rel
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            # Prune empty parent dirs
            self._prune_empty_parents(path.parent)

    def list(self, limit: int = 20) -> list[Snapshot]:
        """List snapshots, newest first."""
        try:
            output = self._git(
                "log", f"--max-count={limit}",
                "--pretty=format:%H%x09%at%x09%s",
                "--no-color",
            )
        except SnapshotError:
            return []

        snapshots = []
        for line in output.split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            if len(parts) < 2:
                continue
            sha = parts[0]
            ts = int(parts[1]) if parts[1].isdigit() else 0
            label = parts[2] if len(parts) > 2 else ""
            snapshots.append(Snapshot(SnapshotId(sha), label, ts))
        return snapshots

    def prune_older_than(self, max_age_days: int = 7) -> int:
        """Remove snapshots older than max_age_days."""
        now = time.time()
        cutoff = now - (max_age_days * 86400)

        snapshots = self.list(limit=10000)
        if not snapshots:
            return 0

        # Find the first snapshot older than cutoff
        cut_index = None
        for i, s in enumerate(snapshots):
            if s.timestamp <= cutoff:
                cut_index = i
                break

        if cut_index is None:
            return 0

        removed = len(snapshots) - cut_index

        if cut_index == 0:
            # All snapshots are old — wipe refs
            refs_dir = self._git_dir / "refs" / "heads"
            if refs_dir.exists():
                for f in refs_dir.iterdir():
                    if f.is_file():
                        f.unlink()
            packed = self._git_dir / "packed-refs"
            if packed.exists():
                packed.unlink()
        else:
            # Reset HEAD to the oldest surviving snapshot
            survivor = snapshots[cut_index - 1]
            self._git("update-ref", "HEAD", survivor.id.sha)

        # Reclaim space
        try:
            self._git("reflog", "expire", "--expire=now", "--all")
            self._git("gc", "--prune=now", "--quiet")
        except SnapshotError:
            pass

        return removed

    # ── Helpers ────────────────────────────────────────────────────

    def _tree_paths(self, treeish: str) -> set[Path]:
        """Get the set of tracked files in a tree-ish."""
        try:
            output = self._git("ls-tree", "-r", "-z", "--name-only", treeish)
        except SnapshotError:
            return set()
        if not output:
            return set()
        return {Path(p) for p in output.split("\0") if p}

    def _prune_empty_parents(self, path: Path) -> None:
        """Remove empty parent directories up to the workspace root."""
        while path and path != self._work_tree:
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent


# ── Convenience ───────────────────────────────────────────────────────

def auto_snapshot(workspace: Path, label: str) -> Optional[SnapshotId]:
    """Take a snapshot with one call. Returns None if git is unavailable."""
    try:
        repo = SnapshotRepo(workspace)
        return repo.snapshot(label)
    except (SnapshotError, FileNotFoundError):
        return None


def auto_restore(workspace: Path, sha: str) -> bool:
    """Restore a snapshot by SHA. Returns True on success."""
    try:
        repo = SnapshotRepo(workspace)
        repo.restore(SnapshotId(sha))
        return True
    except (SnapshotError, FileNotFoundError):
        return False
