"""Language detection and default LSP server registry.

Port of `crates/tui/src/lsp/registry.rs`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional


class Language(Enum):
    RUST = "rust"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    CPP = "cpp"
    C = "c"
    OTHER = "other"

    def as_key(self) -> str:
        return self.value


# Default server commands per language
# (command, [args...])
_DEFAULT_SERVERS: dict[Language, tuple[str, list[str]]] = {
    Language.RUST: ("rust-analyzer", []),
    Language.PYTHON: ("pyright", ["--stdio"]),
    Language.TYPESCRIPT: ("typescript-language-server", ["--stdio"]),
    Language.JAVASCRIPT: ("typescript-language-server", ["--stdio"]),
    Language.GO: ("gopls", []),
    Language.CPP: ("clangd", []),
    Language.C: ("clangd", []),
}


def detect_language(path: Path) -> Language:
    """Detect language from file extension."""
    ext = path.suffix.lower()
    mapping = {
        ".rs": Language.RUST,
        ".py": Language.PYTHON,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".go": Language.GO,
        ".cpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".c": Language.C,
        ".h": Language.CPP,
        ".hpp": Language.CPP,
    }
    return mapping.get(ext, Language.OTHER)


def server_for(lang: Language) -> Optional[tuple[str, list[str]]]:
    """Get the default LSP server command for a language."""
    return _DEFAULT_SERVERS.get(lang)
