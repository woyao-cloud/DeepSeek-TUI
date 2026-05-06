"""LspManager — lifecycle management for LSP servers.

Port of `crates/tui/src/lsp/mod.rs` (LspManager).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import Diagnostic, Severity, DiagnosticBlock
from .registry import Language, detect_language, server_for
from .client import LspTransport, StdioLspTransport

logger = logging.getLogger(__name__)


@dataclass
class LspConfig:
    """[lsp] configuration schema."""
    enabled: bool = True
    poll_after_edit_ms: int = 5000
    max_diagnostics_per_file: int = 20
    include_warnings: bool = False
    servers: dict[str, list[str]] = field(default_factory=dict)

    def resolve_command(self, lang: Language) -> Optional[tuple[str, list[str]]]:
        """Resolve (command, args) for lang, checking overrides first."""
        if lang.as_key() in self.servers:
            parts = self.servers[lang.as_key()]
            if parts:
                return (parts[0], parts[1:])
        return server_for(lang)


class LspManager:
    """Manages LSP server processes and diagnostics collection."""

    def __init__(self, config: LspConfig, workspace: Path) -> None:
        self._config = config
        self._workspace = workspace
        self._transports: dict[Language, LspTransport] = {}
        self._missing_warned: set[Language] = set()

    @classmethod
    def disabled(cls) -> LspManager:
        return cls(LspConfig(enabled=False), Path())

    def config(self) -> LspConfig:
        return self._config

    async def diagnostics_for(self, file: Path, edit_seq: int = 0) -> Optional[DiagnosticBlock]:
        """Get diagnostics for a file after an edit."""
        if not self._config.enabled:
            return None

        lang = detect_language(file)
        if lang == Language.OTHER:
            return None

        try:
            text = file.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug("lsp: read file failed: %s", e)
            return None

        transport = await self._transport_for(lang)
        if transport is None:
            return None

        try:
            items = await asyncio.wait_for(
                transport.diagnostics_for(file, text, self._config.poll_after_edit_ms),
                timeout=(self._config.poll_after_edit_ms + 2000) / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.debug("lsp: diagnostics timed out for %s", file)
            return None
        except Exception as e:
            logger.debug("lsp: diagnostics failed: %s", e)
            return None

        # Filter by severity
        include_warnings = self._config.include_warnings
        filtered = [
            d for d in items
            if d.severity == Severity.ERROR or
               (d.severity == Severity.WARNING and include_warnings)
        ]
        filtered.sort(key=lambda d: d.severity)

        block = DiagnosticBlock(file=file, items=filtered)
        block.truncate(self._config.max_diagnostics_per_file)
        if block.is_empty():
            return None
        return block

    async def _transport_for(self, lang: Language) -> Optional[LspTransport]:
        """Get or lazily spawn a transport for lang."""
        if lang in self._transports:
            return self._transports[lang]

        resolved = self._config.resolve_command(lang)
        if resolved is None:
            return None

        cmd, args = resolved
        try:
            transport = await StdioLspTransport.spawn(cmd, args, lang, self._workspace)
            self._transports[lang] = transport
            return transport
        except FileNotFoundError:
            if lang not in self._missing_warned:
                self._missing_warned.add(lang)
                logger.warning("lsp: server '%s' not found for %s", cmd, lang.as_key())
            return None
        except Exception as e:
            if lang not in self._missing_warned:
                self._missing_warned.add(lang)
                logger.warning("lsp: failed to start '%s' for %s: %s", cmd, lang.as_key(), e)
            return None

    async def shutdown_all(self) -> None:
        """Shut down all spawned LSP servers."""
        for transport in self._transports.values():
            try:
                await transport.shutdown()
            except Exception:
                pass
        self._transports.clear()
