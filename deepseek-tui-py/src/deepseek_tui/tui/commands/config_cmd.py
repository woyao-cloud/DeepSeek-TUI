"""Config commands — config, settings, yolo, agent, plan, trust, logout, lsp, profile."""

from . import CommandResult, CommandInfo, register


def _config_cmd(arg: str | None = None, **kwargs) -> CommandResult:
    return CommandResult.action("open_config")


def _settings(**kwargs) -> CommandResult:
    try:
        from ...config import ConfigStore
        store = ConfigStore.load(None)
        cfg = store.config
        lines = ["Current settings:"]
        lines.append(f"  Provider: {cfg.provider.value if cfg.provider else 'not set'}")
        for k, v in cfg.list_values().items():
            lines.append(f"  {k} = {v}")
        return CommandResult.msg("\n".join(lines))
    except Exception as e:
        return CommandResult.error(str(e))


def _yolo(**kwargs) -> CommandResult:
    return CommandResult.action("toggle_yolo")


def _agent(**kwargs) -> CommandResult:
    return CommandResult.action("toggle_agent")


def _plan(**kwargs) -> CommandResult:
    try:
        from ...tools.todo_plan_tools import _global_plan
        steps = _global_plan.list()
        if not steps:
            return CommandResult.msg("No active plan.")
        lines = ["Current plan:"]
        for i, s in enumerate(steps):
            marker = {"pending": "○", "in_progress": "◉", "completed": "●"}.get(s.get("status", ""), "○")
            lines.append(f"  {marker} {s.get('step', '')}")
        return CommandResult.msg("\n".join(lines))
    except Exception:
        return CommandResult.msg("Plan system not available.")


def _trust(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.msg("Trust settings:\n  Workspace: trusted")
    parts = arg.split()
    match parts[0]:
        case "on":
            return CommandResult.msg("Trust mode enabled.")
        case "off":
            return CommandResult.msg("Trust mode disabled.")
        case "list":
            return CommandResult.msg("Trusted paths:\n  (none configured)")
        case _:
            return CommandResult.error("Usage: /trust [on|off|add <path>|remove <path>|list]")


def _logout(**kwargs) -> CommandResult:
    return CommandResult.action("logout")


def _lsp(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.msg("LSP diagnostics: enabled")
    match arg:
        case "on":
            return CommandResult.msg("LSP diagnostics enabled.")
        case "off":
            return CommandResult.msg("LSP diagnostics disabled.")
        case "status":
            return CommandResult.msg("LSP: enabled (rust-analyzer, pyright)")
        case _:
            return CommandResult.error("Usage: /lsp [on|off|status]")


def _profile(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /profile <name>")
    return CommandResult.action(f"switch_profile {arg}")


def register_all() -> None:
    register(CommandInfo("config", usage="/config",
                         description="Open the configuration editor.",
                         handler=_config_cmd))
    register(CommandInfo("settings", usage="/settings",
                         description="Show current configuration values.",
                         handler=_settings))
    register(CommandInfo("yolo", usage="/yolo",
                         description="Toggle YOLO mode (auto-approve all).",
                         handler=_yolo))
    register(CommandInfo("agent", usage="/agent",
                         description="Toggle agent mode.",
                         handler=_agent))
    register(CommandInfo("plan", usage="/plan",
                         description="Show the current implementation plan.",
                         handler=_plan))
    register(CommandInfo("trust", usage="/trust [on|off|add|remove|list]",
                         description="Manage workspace trust settings.",
                         handler=_trust))
    register(CommandInfo("logout", usage="/logout",
                         description="Log out and clear saved API key.",
                         handler=_logout))
    register(CommandInfo("lsp", usage="/lsp [on|off|status]",
                         description="Manage LSP diagnostics.",
                         handler=_lsp))
    register(CommandInfo("profile", usage="/profile <name>",
                         description="Switch to a different config profile.",
                         handler=_profile))
