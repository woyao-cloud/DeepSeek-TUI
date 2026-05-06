"""Tool commands — review, restore, undo, retry, init, rlm, share."""

from . import CommandResult, CommandInfo, register


def _review(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /review <target>")
    return CommandResult.action(f"review {arg}")


def _restore(arg: str | None = None, **kwargs) -> CommandResult:
    n = arg if arg else "1"
    return CommandResult.action(f"restore {n}")


def _undo(**kwargs) -> CommandResult:
    return CommandResult.action("undo_last_edit")


def _retry(**kwargs) -> CommandResult:
    return CommandResult.action("retry_last_turn")


def _init(**kwargs) -> CommandResult:
    return CommandResult.action("init_project")


def _rlm(arg: str | None = None, **kwargs) -> CommandResult:
    if not arg:
        return CommandResult.error("Usage: /rlm <prompt>")
    if len(arg) < 50:
        return CommandResult.msg(
            "Tip: RLM is designed for processing LONG prompts (>100 chars). "
            "For short queries, just type the message directly."
        )
    return CommandResult.action(f"rlm {arg}")


def _share(arg: str | None = None, **kwargs) -> CommandResult:
    return CommandResult.action("share_conversation")


def register_all() -> None:
    register(CommandInfo("review", usage="/review <target>",
                         description="Run a code review over the specified target.",
                         handler=_review))
    register(CommandInfo("restore", usage="/restore [N]",
                         description="Restore workspace to a previous snapshot.",
                         handler=_restore))
    register(CommandInfo("init", usage="/init",
                         description="Create a default AGENTS.md in the current directory.",
                         handler=_init))
    register(CommandInfo("rlm", aliases=["recursive"], usage="/rlm <prompt>",
                         description="Process a prompt using Recursive Language Model.",
                         handler=_rlm))
    register(CommandInfo("share", usage="/share",
                         description="Share the current conversation via link.",
                         handler=_share))
