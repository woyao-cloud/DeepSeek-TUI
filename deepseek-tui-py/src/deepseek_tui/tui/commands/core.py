"""Core commands — help, clear, exit, model, models, provider, links, home, subagents."""

from . import CommandResult, CommandInfo, register


def _help(arg: str | None = None, **kwargs) -> CommandResult:
    if arg:
        from . import get_registry
        cmd = get_registry().get(arg)
        if cmd:
            desc = cmd.description
            aliases = f"  aliases: {', '.join(cmd.aliases)}" if cmd.aliases else ""
            return CommandResult.msg(
                f"/{cmd.name} — {desc}\n"
                f"  Usage: {cmd.usage}{aliases}"
            )
        return CommandResult.error(f"No such command: /{arg}")
    lines = ["Available commands:", ""]
    from . import get_registry
    for cmd in get_registry().all():
        lines.append(f"  /{cmd.name:<12} {cmd.description}")
    lines.append("")
    lines.append("Type /help <command> for details on a specific command.")
    return CommandResult.msg("\n".join(lines))


def _clear(**kwargs) -> CommandResult:
    return CommandResult.action("clear")


def _exit(**kwargs) -> CommandResult:
    return CommandResult.action("quit")


def _model(arg: str | None = None, **kwargs) -> CommandResult:
    if arg:
        from ...config import ConfigStore
        store = ConfigStore.load(None)
        store.config.set_value("model", arg)
        store.save()
        return CommandResult.msg(f"Model set to: {arg}")
    from ...config import ConfigStore, CliRuntimeOverrides
    store = ConfigStore.load(None)
    resolved = store.config.resolve_runtime_options(CliRuntimeOverrides())
    return CommandResult.msg(f"Current model: {resolved.model}")


def _models(**kwargs) -> CommandResult:
    from ...agent import ModelRegistry
    registry = ModelRegistry()
    lines = ["Available models:"]
    for m in registry.list():
        lines.append(f"  {m.id} ({m.provider.value})")
    return CommandResult.msg("\n".join(lines))


def _links(**kwargs) -> CommandResult:
    return CommandResult.msg(
        "DeepSeek Links:\n"
        "  Platform: https://platform.deepseek.com\n"
        "  API Docs: https://api-docs.deepseek.com\n"
        "  Dashboard: https://platform.deepseek.com/usage\n"
        "  GitHub: https://github.com/deepseek-ai"
    )


def _home(**kwargs) -> CommandResult:
    return CommandResult.msg(
        "DeepSeek TUI v0.1.0\n"
        "  Type /help for commands\n"
        "  Type /model to see current model\n"
        "  Type /links for DeepSeek resources"
    )


def _subagents(**kwargs) -> CommandResult:
    """List running sub-agents."""
    try:
        from ...tools.subagent import _agent_manager
        agents = _agent_manager.list()
        if not agents:
            return CommandResult.msg("No running sub-agents.")
        lines = ["Sub-agents:"]
        for a in agents:
            lines.append(f"  [{a.status}] {a.id} — {a.name}")
        return CommandResult.msg("\n".join(lines))
    except Exception:
        return CommandResult.msg("Sub-agent system not available.")


def register_all() -> None:
    register(CommandInfo("help", aliases=["?"], usage="/help [command]",
                         description="Show this help message or details for a specific command.",
                         handler=_help))
    register(CommandInfo("clear", usage="/clear",
                         description="Clear the conversation transcript.",
                         handler=_clear))
    register(CommandInfo("exit", aliases=["quit", "q"], usage="/exit",
                         description="Exit the application.",
                         handler=_exit))
    register(CommandInfo("model", usage="/model [name]",
                         description="Show or set the current model.",
                         handler=_model))
    register(CommandInfo("models", usage="/models",
                         description="List available models from the registry.",
                         handler=_models))
    register(CommandInfo("links", aliases=["dashboard", "api"], usage="/links",
                         description="Show DeepSeek platform links.",
                         handler=_links))
    register(CommandInfo("home", aliases=["stats", "overview"], usage="/home",
                         description="Show the TUI home/dashboard view.",
                         handler=_home))
    register(CommandInfo("subagents", aliases=["agents"], usage="/subagents",
                         description="List running sub-agents.",
                         handler=_subagents))
