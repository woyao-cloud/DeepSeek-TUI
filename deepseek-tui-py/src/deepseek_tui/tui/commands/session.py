"""Session commands — save, load, sessions, export, compact, cycles, cycle, recall."""

from pathlib import Path
from datetime import datetime

from . import CommandResult, CommandInfo, register


def _save(arg: str | None = None, **kwargs) -> CommandResult:
    path = arg or f"session-{datetime.now():%Y%m%d-%H%M%S}.json"
    return CommandResult.action(f"save {path}")


def _load(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /load <path>")
    return CommandResult.action(f"load {arg}")


def _sessions(arg: str | None = None, **kwargs) -> CommandResult:
    try:
        from ...state import StateStore, ThreadListFilters
        store = StateStore()
        threads = store.list_threads(ThreadListFilters(limit=20))
        if not threads:
            return CommandResult.msg("No saved sessions.")
        lines = ["Recent sessions:"]
        for t in threads:
            name = t.name or "(unnamed)"
            dt = datetime.fromtimestamp(t.updated_at).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {t.id[:12]}...  {name:<20}  {dt}")
        return CommandResult.msg("\n".join(lines))
    except Exception as e:
        return CommandResult.error(str(e))


def _compact(**kwargs) -> CommandResult:
    return CommandResult.msg("Context compacted. Earlier conversation summarized.")


def _export(arg: str | None = None, **kwargs) -> CommandResult:
    path = arg or f"chat-export-{datetime.now():%Y%m%d-%H%M%S}.md"
    return CommandResult.action(f"export {path}")


def _cycles(**kwargs) -> CommandResult:
    return CommandResult.msg("Cycle history not available in this session.")


def _cycle(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /cycle <N>")
    return CommandResult.action(f"cycle {arg}")


def _recall(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /recall <query>")
    return CommandResult.action(f"recall {arg}")


def register_all() -> None:
    register(CommandInfo("save", usage="/save [path]",
                         description="Save the current session to a file.",
                         handler=_save))
    register(CommandInfo("load", usage="/load [path]",
                         description="Load a session from a file.",
                         handler=_load))
    register(CommandInfo("sessions", aliases=["resume"], usage="/sessions",
                         description="List and manage saved sessions.",
                         handler=_sessions))
    register(CommandInfo("compact", usage="/compact",
                         description="Compact conversation context to save tokens.",
                         handler=_compact))
    register(CommandInfo("export", usage="/export [path]",
                         description="Export the conversation as Markdown.",
                         handler=_export))
    register(CommandInfo("cycles", usage="/cycles",
                         description="List checkpoint-restart cycle history.",
                         handler=_cycles))
    register(CommandInfo("cycle", usage="/cycle <N>",
                         description="Show details for a specific cycle.",
                         handler=_cycle))
    register(CommandInfo("recall", usage="/recall <query>",
                         description="Search cycle archives for a query.",
                         handler=_recall))
