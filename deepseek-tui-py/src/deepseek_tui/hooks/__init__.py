"""Hooks system — port of `deepseek-hooks` crate.

Lifecycle hooks for tool events (stdout, jsonl, webhook).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class HookEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class HookSink:
    """Abstract hook sink."""
    def emit(self, event: HookEvent) -> None:
        raise NotImplementedError


class StdoutHookSink(HookSink):
    """Print events to stdout."""
    def emit(self, event: HookEvent) -> None:
        print(f"[hook] {event.kind}: {json.dumps(event.payload, default=str)[:200]}")


class JsonlHookSink(HookSink):
    """Write events to a JSONL file."""
    def __init__(self, path: str) -> None:
        self._path = path

    def emit(self, event: HookEvent) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": event.kind,
                "payload": event.payload,
            }
            f.write(json.dumps(record, default=str) + "\n")


class HookDispatcher:
    """Dispatch hook events to all registered sinks."""

    def __init__(self) -> None:
        self._sinks: list[HookSink] = []

    def add_sink(self, sink: HookSink) -> None:
        self._sinks.append(sink)

    async def emit(self, kind: str, payload: Optional[dict[str, Any]] = None) -> None:
        event = HookEvent(kind=kind, payload=payload or {})
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                pass  # Don't let hooks crash the engine
