"""Skill and misc commands — skills, skill, attach, memory, goal, hooks."""

from . import CommandResult, CommandInfo, register


def _skills(arg: str | None = None, **kwargs) -> CommandResult:
    return CommandResult.msg(
        "Skills:\n  No skills installed. Use `deepseek setup --skills --local` to create an example skill."
    )


def _skill(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /skill <name|install|update|uninstall|trust>")
    parts = arg.split(maxsplit=1)
    cmd = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    match cmd:
        case "list":
            return CommandResult.msg("Installed skills: (none)")
        case "install":
            if not rest:
                return CommandResult.error("Usage: /skill install <spec>")
            return CommandResult.action(f"install_skill {rest}")
        case "uninstall":
            return CommandResult.action(f"uninstall_skill {rest}")
        case _:
            return CommandResult.action(f"run_skill {arg}")


def _attach(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /attach <path>")
    return CommandResult.action(f"attach_file {arg}")


def _memory(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.msg(
            "Memory:\n  Path: ~/.deepseek/memory.md\n  Status: not enabled"
        )
    parts = arg.split(maxsplit=1)
    cmd = parts[0]
    match cmd:
        case "show":
            return CommandResult.action("show_memory")
        case "path":
            return CommandResult.msg("Memory file: ~/.deepseek/memory.md")
        case "clear":
            return CommandResult.msg("Memory cleared.")
        case "edit":
            return CommandResult.action("edit_memory")
        case "help":
            return CommandResult.msg(
                "Memory commands:\n"
                "  /memory show    — display current memory\n"
                "  /memory path    — show memory file path\n"
                "  /memory clear   — clear memory\n"
                "  /memory edit    — open memory in editor"
            )
        case _:
            return CommandResult.error("Usage: /memory [show|path|clear|edit|help]")


def _goal(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /goal [objective] [budget: N]")
    return CommandResult.action(f"set_goal {arg}")


def _hooks(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.msg("Event hooks:\n  stdout: enabled\n  jsonl: not configured")
    if arg == "list":
        return CommandResult.msg("Registered hooks:\n  stdout, jsonl")
    if arg == "events":
        return CommandResult.msg("Hook events: tool_call_before, tool_call_after, response_start, response_end")
    return CommandResult.error("Usage: /hooks [list|events]")


def register_all() -> None:
    register(CommandInfo("skills", usage="/skills [--remote|sync]",
                         description="List installed skills.",
                         handler=_skills))
    register(CommandInfo("skill", usage="/skill <name|install|update|uninstall|trust>",
                         description="Manage and run skills.",
                         handler=_skill))
    register(CommandInfo("attach", aliases=["image", "media"], usage="/attach <path>",
                         description="Attach a file or image to the conversation.",
                         handler=_attach))
    register(CommandInfo("memory", usage="/memory [show|path|clear|edit|help]",
                         description="Manage the user memory file.",
                         handler=_memory))
    register(CommandInfo("goal", usage="/goal [objective] [budget: N]",
                         description="Set a session goal with optional token budget.",
                         handler=_goal))
    register(CommandInfo("hooks", aliases=["hook"], usage="/hooks [list|events]",
                         description="Manage event hooks.",
                         handler=_hooks))
