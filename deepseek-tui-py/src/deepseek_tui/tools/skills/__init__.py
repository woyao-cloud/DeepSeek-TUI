"""Skills system — SKILL.md parsing, discovery, and registry.

Port of `crates/tui/src/skills/mod.rs`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """A parsed skill from a SKILL.md file."""
    name: str
    description: str
    body: str
    path: Path = Path()
    allowed_tools: list[str] = field(default_factory=list)


class SkillRegistry:
    """Collection of discovered skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._warnings: list[str] = []

    @classmethod
    def discover(cls, directory: Path) -> SkillRegistry:
        """Discover skills from a directory."""
        registry = cls()
        if not directory.exists():
            return registry

        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                skill = cls._parse_skill(skill_file, content)
                if skill and skill.name:
                    skill.path = skill_file
                    registry._skills[skill.name] = skill
                else:
                    registry._warnings.append(
                        f"Failed to parse {skill_file}: missing 'name' in frontmatter"
                    )
            except Exception as e:
                registry._warnings.append(
                    f"Failed to read {skill_file}: {e}"
                )
        return registry

    @staticmethod
    def _parse_skill(path: Path, content: str) -> Optional[Skill]:
        """Parse a SKILL.md file with YAML frontmatter."""
        trimmed = content.strip()
        if not trimmed.startswith("---"):
            return None

        # Extract frontmatter between --- delimiters
        parts = trimmed.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter_raw = parts[1]
        body = parts[2].strip()

        # Parse simple key: value pairs from frontmatter
        metadata: dict[str, str] = {}
        allowed_tools: list[str] = []
        for line in frontmatter_raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                metadata[key] = value
                if key == "allowed-tools":
                    allowed_tools = [t.strip() for t in value.split(",") if t.strip()]

        name = metadata.get("name", "")
        if not name:
            return None

        description = metadata.get("description", "")
        return Skill(
            name=name,
            description=description,
            body=body,
            allowed_tools=allowed_tools,
        )

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    @property
    def warnings(self) -> list[str]:
        return self._warnings

    def is_empty(self) -> bool:
        return len(self._skills) == 0

    def __len__(self) -> int:
        return len(self._skills)


def default_skills_dir() -> Path:
    """Get the default skills directory (~/.deepseek/skills/)."""
    return Path.home() / ".deepseek" / "skills"


def discover_in_workspace(workspace: Path) -> SkillRegistry:
    """Discover skills from all candidate directories in a workspace."""
    merged = SkillRegistry()
    candidates = [
        workspace / ".agents" / "skills",
        workspace / "skills",
        workspace / ".opencode" / "skills",
        workspace / ".claude" / "skills",
        default_skills_dir(),
    ]
    for directory in candidates:
        if directory.exists():
            registry = SkillRegistry.discover(directory)
            for skill in registry.list():
                if merged.get(skill.name) is None:
                    merged._skills[skill.name] = skill
            for warning in registry.warnings:
                merged._warnings.append(warning)
    return merged
