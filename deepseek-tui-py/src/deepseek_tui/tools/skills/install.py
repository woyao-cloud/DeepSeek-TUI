"""Skills installation — community registry / install / sync.

Port of `crates/tui/src/skills/install.rs`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from . import Skill, default_skills_dir

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/Hmbown/deepseek-skills/main/registry.json"
DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MiB


class InstallError(Exception):
    pass


async def fetch_registry(url: Optional[str] = None) -> dict:
    """Fetch the skill registry index."""
    registry_url = url or DEFAULT_REGISTRY_URL
    try:
        with urlopen(registry_url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise InstallError(f"Failed to fetch registry: {e}")


async def install_skill(
    spec: str,
    skills_dir: Optional[Path] = None,
    max_size: int = DEFAULT_MAX_SIZE_BYTES,
) -> Skill:
    """Install a skill by spec (name or URL)."""
    target_dir = skills_dir or default_skills_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    if spec.startswith("http://") or spec.startswith("https://"):
        return await _install_from_url(spec, target_dir, max_size)

    # Look up in registry
    registry = await fetch_registry()
    entries = registry.get("skills", [])
    for entry in entries:
        if entry.get("name") == spec or entry.get("id") == spec:
            url = entry.get("url") or entry.get("archive_url")
            if url:
                return await _install_from_url(url, target_dir, max_size)
            raise InstallError(f"Skill '{spec}' has no download URL in registry")
    raise InstallError(f"Skill '{spec}' not found in registry")


async def _install_from_url(url: str, target_dir: Path, max_size: int) -> Skill:
    """Download and install a skill from a URL."""
    import asyncio
    import aiohttp

    # Use aiohttp if available, otherwise fallback to urllib
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                content = await resp.read()
    except ImportError:
        # Fallback to synchronous urllib
        with urlopen(url, timeout=30) as resp:
            content = resp.read()

    if len(content) > max_size:
        raise InstallError(f"Skill archive too large ({len(content)} > {max_size} bytes)")

    # Handle zip archives
    if url.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "skill.zip"
            zip_path.write_bytes(content)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
    else:
        # Assume it's a single SKILL.md
        name = url.rsplit("/", 1)[-1].replace(".md", "").replace("-", "_")
        skill_dir = target_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_bytes(content)

    # Re-discover to return the parsed skill
    registry = type("_", (), {"discover": SkillRegistry.discover})().discover(target_dir)
    for skill in registry.list():
        return skill
    raise InstallError("Installation succeeded but skill could not be parsed")
