"""Register all command modules."""
from . import core, session, config_cmd, management, debug, tool_cmds, skills_cmds


def register_all() -> None:
    """Register all slash command handlers from every module."""
    core.register_all()
    session.register_all()
    config_cmd.register_all()
    management.register_all()
    debug.register_all()
    tool_cmds.register_all()
    skills_cmds.register_all()
