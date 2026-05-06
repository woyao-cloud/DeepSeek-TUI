"""Debug commands — cost, tokens, cache, system, context, edit, diff."""

from . import CommandResult, CommandInfo, register


def _cost(**kwargs) -> CommandResult:
    return CommandResult.msg(
        "Session cost:\n  Input tokens: N/A\n  Output tokens: N/A\n  Total cost: N/A"
    )


def _tokens(**kwargs) -> CommandResult:
    return CommandResult.msg(
        "Token usage:\n  Context window: 1,000,000 tokens\n  Current usage: N/A"
    )


def _cache(arg: str | None = None, **kwargs) -> CommandResult:
    count = arg if arg else "10"
    return CommandResult.msg(f"Cache telemetry (last {count} turns): not available.")


def _system(**kwargs) -> CommandResult:
    return CommandResult.action("show_system_prompt")


def _context(**kwargs) -> CommandResult:
    return CommandResult.action("open_context_inspector")


def _edit(**kwargs) -> CommandResult:
    return CommandResult.msg("Edit mode: open an editor for the current input.")


def _diff(arg: str | None = None, **kwargs) -> CommandResult:
    if arg:
        return CommandResult.action(f"show_diff {arg}")
    return CommandResult.action("show_diff")


def _undo(**kwargs) -> CommandResult:
    return CommandResult.action("undo")


def _retry(**kwargs) -> CommandResult:
    return CommandResult.action("retry_last")


def _statusline(**kwargs) -> CommandResult:
    return CommandResult.msg("Status line items: model, cost, tokens, mode, timer")


def register_all() -> None:
    register(CommandInfo("cost", usage="/cost",
                         description="Show token usage and cost for the session.",
                         handler=_cost))
    register(CommandInfo("tokens", usage="/tokens",
                         description="Show current token usage.",
                         handler=_tokens))
    register(CommandInfo("cache", usage="/cache [count]",
                         description="Show prefix-cache hit rate telemetry.",
                         handler=_cache))
    register(CommandInfo("system", usage="/system",
                         description="Show the current system prompt.",
                         handler=_system))
    register(CommandInfo("context", aliases=["ctx"], usage="/context",
                         description="Open the context inspector.",
                         handler=_context))
    register(CommandInfo("edit", usage="/edit",
                         description="Open an external editor for the current input.",
                         handler=_edit))
    register(CommandInfo("diff", usage="/diff [path]",
                         description="Show diff of the working tree.",
                         handler=_diff))
    register(CommandInfo("undo", usage="/undo",
                         description="Undo the last change.",
                         handler=_undo))
    register(CommandInfo("retry", usage="/retry",
                         description="Retry the last LLM request.",
                         handler=_retry))
    register(CommandInfo("statusline", aliases=["status"], usage="/statusline",
                         description="Show or configure status line items.",
                         handler=_statusline))
