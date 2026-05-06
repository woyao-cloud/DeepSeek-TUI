"""User memory file — persistent cross-session notes.

Port of `crates/tui/src/memory.rs`.

Stores a Markdown file at `~/.deepseek/memory.md` (configurable).
The model sees its content as a `<user_memory>` block in the system prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MAX_MEMORY_SIZE = 100 * 1024  # 100 KiB


def load(path: Path) -> Optional[str]:
    """Read the user memory file. Returns None if missing or empty."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if content.strip():
        return content
    return None


def as_system_block(content: str, source: Path) -> Optional[str]:
    """Wrap memory content in a `<user_memory>` block.

    Returns None for empty/whitespace content.
    """
    trimmed = content.strip()
    if not trimmed:
        return None

    payload = trimmed
    if len(content) > MAX_MEMORY_SIZE:
        head = content[:MAX_MEMORY_SIZE]
        payload = head + "\n…(truncated, raise memory.max_size or trim memory.md)"

    return f"<user_memory source=\"{source}\">\n{payload}\n</user_memory>"


def compose_block(enabled: bool, path: Path) -> Optional[str]:
    """Compose `<user_memory>` block for the system prompt.

    Returns None when disabled or file is missing/empty.
    """
    if not enabled:
        return None
    content = load(path)
    if content is None:
        return None
    return as_system_block(content, path)


def append_entry(path: Path, entry: str) -> None:
    """Append a timestamped bullet to the memory file.

    Creates the file (and parent dir) if needed.
    Strips leading `#` from quick-add entries like `# foo`.
    """
    trimmed = entry.lstrip("#").strip()
    if not trimmed:
        raise ValueError("memory entry is empty after stripping `#` prefix")

    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- ({timestamp}) {trimmed}\n")


def default_memory_path() -> Path:
    """Get the default memory file path."""
    return Path.home() / ".deepseek" / "memory.md"
