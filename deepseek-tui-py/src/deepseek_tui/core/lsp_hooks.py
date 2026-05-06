"""LSP engine hooks — called after successful edits.

Port of `crates/tui/src/core/engine/lsp_hooks.rs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..tools.lsp import LspManager
from ..tools.lsp.models import DiagnosticBlock, render_blocks


async def run_post_edit_lsp_hook(
    lsp_manager: LspManager,
    file_path: Path,
    edit_seq: int,
) -> Optional[str]:
    """Run LSP diagnostics after a file edit.

    Returns a rendered diagnostic block string if there are diagnostics,
    or None if clean / LSP is disabled / no server for this language.
    """
    block = await lsp_manager.diagnostics_for(file_path, edit_seq)
    if block is None:
        return None
    return render_blocks([block])


async def flush_pending_lsp_diagnostics(
    lsp_manager: LspManager,
    pending: list[tuple[Path, int]],
) -> Optional[str]:
    """Flush multiple pending LSP diagnostics and return a combined block."""
    blocks: list[DiagnosticBlock] = []
    for file_path, edit_seq in pending:
        block = await lsp_manager.diagnostics_for(file_path, edit_seq)
        if block is not None:
            blocks.append(block)
    if not blocks:
        return None
    return render_blocks(blocks)
