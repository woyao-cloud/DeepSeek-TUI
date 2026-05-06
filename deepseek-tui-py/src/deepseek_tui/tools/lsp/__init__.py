"""LSP module — port of `crates/tui/src/lsp/`.

Post-edit diagnostics injection via Language Server Protocol.
"""

from .models import Diagnostic, Severity, DiagnosticBlock, render_blocks
from .registry import Language, detect_language, server_for
from .client import LspTransport, StdioLspTransport
from .manager import LspManager, LspConfig

__all__ = [
    "Diagnostic", "Severity", "DiagnosticBlock", "render_blocks",
    "Language", "detect_language", "server_for",
    "LspTransport", "StdioLspTransport",
    "LspManager", "LspConfig",
]
