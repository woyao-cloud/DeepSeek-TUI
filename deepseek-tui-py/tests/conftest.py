"""Shared test configuration."""

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The Windows user temp directory may be unavailable in the sandboxed test
# environment. Keep tempfile-based tests inside the writable workspace area.
_repo_root = Path(__file__).resolve().parents[1]
_test_tmp = Path(os.environ.get("DEEPSEEK_TUI_TEST_TMP", str(_repo_root / "test-tmp")))
_test_tmp.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(_test_tmp)
os.environ["TEMP"] = str(_test_tmp)
os.environ["HOME"] = str(_repo_root / "test-home")
os.environ["USERPROFILE"] = str(_repo_root / "test-home")
tempfile.tempdir = str(_test_tmp)
Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)


class _WorkspaceTemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
        base = Path(dir) if dir else _test_tmp
        name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
        self.name = str(base / name)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        Path(self.name).mkdir(parents=True, exist_ok=False)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory

# Enable asyncio mode for pytest
import pytest
pytest.register_assert_rewrite("deepseek_tui")
