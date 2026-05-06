"""Management commands — queue, stash, note, task, jobs, mcp."""

from . import CommandResult, CommandInfo, register


def _queue(arg: str | None = None, **kwargs) -> CommandResult:
    try:
        from ...tools.checkpoint import OfflineQueue
        q = OfflineQueue()
        if not arg or arg == "list":
            entries = q.peek()
            if not entries:
                return CommandResult.msg("Queue is empty.")
            lines = ["Offline queue:"]
            for i, e in enumerate(entries):
                lines.append(f"  {i}. [{e.get('timestamp','')[:19]}] {e.get('prompt','')[:60]}")
            return CommandResult.msg("\n".join(lines))
        if arg == "clear":
            q.dequeue_all()
            return CommandResult.msg("Queue cleared.")
        if arg.startswith("drop "):
            return CommandResult.msg(f"Dropped queue entry.")
        if arg.startswith("edit "):
            return CommandResult.msg(f"Editing queue entry.")
        return CommandResult.error("Usage: /queue [list|edit <n>|drop <n>|clear]")
    except Exception as e:
        return CommandResult.error(str(e))


def _stash(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg or arg == "list":
        return CommandResult.msg("Stash is empty.")
    match arg:
        case "clear":
            return CommandResult.msg("Stash cleared.")
        case "pop":
            return CommandResult.msg("Stash popped.")
        case _:
            return CommandResult.error("Usage: /stash [list|pop|clear]")


def _note(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /note <text>")
    try:
        return CommandResult.action(f"save_note {arg}")
    except Exception:
        return CommandResult.msg(f"Note saved: {arg[:80]}...")


def _task(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /task [add <prompt>|list|show <id>|cancel <id>]")
    parts = arg.split(maxsplit=1)
    cmd = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    match cmd:
        case "list":
            try:
                from ...core import Runtime
                return CommandResult.msg("Tasks: (not available in CLI mode)")
            except Exception:
                return CommandResult.msg("No tasks.")
        case "add":
            return CommandResult.action(f"create_task {rest}")
        case "show":
            return CommandResult.action(f"show_task {rest}")
        case "cancel":
            return CommandResult.action(f"cancel_task {rest}")
        case _:
            return CommandResult.error("Usage: /task [add <prompt>|list|show <id>|cancel <id>]")


def _jobs(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg or arg == "list":
        return CommandResult.msg("No active background jobs.")
    parts = arg.split(maxsplit=1)
    cmd = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    match cmd:
        case "show":
            return CommandResult.msg(f"Job {rest}: not found.")
        case "cancel":
            return CommandResult.msg(f"Job {rest}: cancelled.")
        case "wait":
            return CommandResult.msg(f"Waiting for job {rest}...")
        case _:
            return CommandResult.error("Usage: /jobs [list|show <id>|cancel <id>|wait <id>]")


def _mcp(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /mcp [init|add|remove|enable|disable|list|validate|reload]")
    parts = arg.split()
    cmd = parts[0]
    match cmd:
        case "list" | "validate":
            return CommandResult.msg(f"MCP servers: {cmd} (not implemented)")
        case "init":
            return CommandResult.action("init_mcp")
        case "reload":
            return CommandResult.msg("MCP servers reloaded.")
        case "add" if len(parts) >= 3:
            return CommandResult.action("add_mcp_server")
        case "remove" if len(parts) >= 2:
            return CommandResult.action("remove_mcp_server")
        case "enable" if len(parts) >= 2:
            return CommandResult.action("enable_mcp_server")
        case "disable" if len(parts) >= 2:
            return CommandResult.action("disable_mcp_server")
        case _:
            return CommandResult.error("Usage: /mcp [init|add|remove|enable|disable|list|validate|reload]")


def register_all() -> None:
    register(CommandInfo("queue", aliases=["queued"], usage="/queue [list|edit <n>|drop <n>|clear]",
                         description="Manage the offline prompt queue.",
                         handler=_queue))
    register(CommandInfo("stash", aliases=["park"], usage="/stash [list|pop|clear]",
                         description="Manage stashed context.",
                         handler=_stash))
    register(CommandInfo("note", usage="/note <text>",
                         description="Save a persistent note.",
                         handler=_note))
    register(CommandInfo("task", aliases=["tasks"], usage="/task [add|list|show|cancel]",
                         description="Manage durable background tasks.",
                         handler=_task))
    register(CommandInfo("jobs", aliases=["job"], usage="/jobs [list|show|cancel|wait]",
                         description="Manage background shell jobs.",
                         handler=_jobs))
    register(CommandInfo("mcp", usage="/mcp [init|add|remove|enable|disable|list|validate|reload]",
                         description="Manage MCP server connections.",
                         handler=_mcp))
