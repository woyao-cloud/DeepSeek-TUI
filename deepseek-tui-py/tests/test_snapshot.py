"""Tests for the snapshot/revert_turn system."""

import os
import tempfile
from pathlib import Path

import pytest

from deepseek_tui.tools.snapshot import (
    SnapshotRepo, SnapshotId, Snapshot,
    snapshot_dir_for, snapshot_git_dir,
    auto_snapshot, auto_restore,
    SnapshotError,
)


class TestSnapshotPaths:
    def test_snapshot_dir_under_deepseek(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "project"
            workspace.mkdir()
            snap_dir = snapshot_dir_for(workspace, home=Path(tmp))
            # Path should be: <tmp>/.deepseek/snapshots/<hash>/<hash>
            rel = snap_dir.relative_to(Path(tmp))
            parts = list(rel.parts)
            assert parts[0] == ".deepseek"
            assert parts[1] == "snapshots"
            assert len(parts) == 4  # .deepseek/snapshots/<proj>/<wt>

    def test_git_dir_appends_dot_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "project"
            workspace.mkdir()
            git_dir = snapshot_git_dir(workspace)
            assert git_dir.name == ".git"

    def test_snapshot_dir_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "project"
            workspace.mkdir()
            d1 = snapshot_dir_for(workspace, home=Path(tmp))
            d2 = snapshot_dir_for(workspace, home=Path(tmp))
            assert d1 == d2


class TestSnapshotRepo:
    @pytest.fixture
    def repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            repo = SnapshotRepo(workspace)
            yield repo

    def test_init_creates_git_dir(self, repo):
        assert repo.git_dir.exists()
        assert (repo.git_dir / "HEAD").exists()

    def test_snapshot_creates_commit(self, repo):
        # Write a file and snapshot
        (repo.work_tree / "test.txt").write_text("hello")
        sid = repo.snapshot("pre-turn:1")
        assert len(sid.sha) == 40  # Full SHA

        snapshots = repo.list()
        assert len(snapshots) == 1
        assert snapshots[0].label == "pre-turn:1"

    def test_restore_reverts_file(self, repo):
        f = repo.work_tree / "file.txt"
        f.write_text("original")
        sid = repo.snapshot("pre-turn:1")

        f.write_text("clobbered")
        repo.snapshot("post-turn:1")

        repo.restore(sid)
        assert f.read_text() == "original"

    def test_restore_removes_added_files(self, repo):
        original = repo.work_tree / "original.txt"
        added = repo.work_tree / "added.txt"

        original.write_text("stay")
        sid = repo.snapshot("pre-turn:1")

        added.write_text("new file")
        repo.snapshot("post-turn:1")

        repo.restore(sid)
        assert original.exists()
        assert not added.exists()

    def test_list_respects_limit(self, repo):
        for i in range(5):
            (repo.work_tree / "f.txt").write_text(f"v{i}")
            repo.snapshot(f"turn:{i}")

        three = repo.list(limit=3)
        assert len(three) == 3
        # Newest first
        assert three[0].label == "turn:4"

    def test_multiple_snapshots_in_order(self, repo):
        for i in range(3):
            (repo.work_tree / "data.txt").write_text(f"content {i}")
            repo.snapshot(f"turn:{i}")

        all_snaps = repo.list(limit=10)
        assert len(all_snaps) == 3
        # Newest first
        assert all_snaps[0].label == "turn:2"
        assert all_snaps[2].label == "turn:0"

    def test_no_snapshots_returns_empty(self, repo):
        assert repo.list() == []

    def test_auto_snapshot_and_restore(self, repo):
        f = repo.work_tree / "test.txt"
        f.write_text("auto-test")
        sid = auto_snapshot(repo.work_tree, "auto:1")
        assert sid is not None

        f.write_text("modified")
        ok = auto_restore(repo.work_tree, sid.sha)
        assert ok
        assert f.read_text() == "auto-test"

    def test_snapshot_does_not_create_user_git(self, repo):
        repo.snapshot("test")
        # The workspace should NOT have a .git dir
        assert not (repo.work_tree / ".git").exists()
        # The side repo should
        assert repo.git_dir.exists()

    def test_restore_nonexistent_snapshot_raises(self, repo):
        bad_id = SnapshotId("0" * 40)
        with pytest.raises(SnapshotError):
            repo.restore(bad_id)


class TestRevertTurnTool:
    @pytest.fixture
    def ctx(self):
        from deepseek_tui.tools import ToolContext, ToolParams
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            yield ToolContext(params=ToolParams(workspace=workspace, cwd=workspace))

    async def test_revert_tool_no_snapshots(self, ctx):
        from deepseek_tui.tools.revert_turn import RevertTurnTool
        tool = RevertTurnTool()
        result = await tool.execute({"n": 1}, ctx)
        assert not result.success
        assert "No snapshots" in result.content

    async def test_revert_tool_success(self, ctx):
        from deepseek_tui.tools.revert_turn import RevertTurnTool
        from deepseek_tui.tools.snapshot import SnapshotRepo

        # Create a snapshot first
        repo = SnapshotRepo(ctx.params.workspace)
        f = ctx.params.workspace / "test.txt"
        f.write_text("original")
        repo.snapshot("test")

        # Modify
        f.write_text("changed")

        # Revert
        tool = RevertTurnTool()
        result = await tool.execute({"n": 1}, ctx)
        assert result.success
        assert f.read_text() == "original"

    async def test_pre_post_turn_snapshots(self, ctx):
        from deepseek_tui.tools.revert_turn import pre_turn_snapshot, post_turn_snapshot

        ws = ctx.params.workspace
        (ws / "file.txt").write_text("v1")
        await pre_turn_snapshot(ws, 1)

        (ws / "file.txt").write_text("v2")
        await post_turn_snapshot(ws, 1)

        # Verify snapshots exist
        repo = SnapshotRepo(ws)
        snaps = repo.list()
        assert len(snaps) >= 2
        labels = [s.label for s in snaps]
        assert "pre-turn:1" in labels
        assert "post-turn:1" in labels
