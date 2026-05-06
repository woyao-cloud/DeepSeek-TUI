"""Default built-in tool registration."""

from __future__ import annotations

from .file_tools import ApplyPatchTool, EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from .registry import ToolRegistry
from .remember_tool import RememberTool
from .revert_turn import RevertTurnTool
from .rlm_tool import RlmTool
from .search_git_tools import (
    FileSearchTool,
    GitBlameTool,
    GitDiffTool,
    GitLogTool,
    GitShowTool,
    GitStatusTool,
    GrepFilesTool,
)
from .shell_tools import ExecShellTool, NoteTool, ShellInteractTool, ShellWaitTool
from .todo_plan_tools import (
    DiagnosticsTool,
    TodoAddTool,
    TodoListTool,
    TodoUpdateTool,
    TodoWriteTool,
    UpdatePlanTool,
)
from .web_validation_tools import (
    FetchUrlTool,
    RequestUserInputTool,
    ValidateDataTool,
    WebSearchTool,
)


def build_default_tool_registry() -> ToolRegistry:
    """Build the default registry used by CLI, TUI, and server runtimes."""
    registry = ToolRegistry()
    for tool in [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirTool(),
        ApplyPatchTool(),
        ExecShellTool(),
        ShellWaitTool(),
        ShellInteractTool(),
        NoteTool(),
        GrepFilesTool(),
        FileSearchTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
        GitShowTool(),
        GitBlameTool(),
        TodoWriteTool(),
        TodoAddTool(),
        TodoUpdateTool(),
        TodoListTool(),
        UpdatePlanTool(),
        DiagnosticsTool(),
        ValidateDataTool(),
        WebSearchTool(),
        FetchUrlTool(),
        RequestUserInputTool(),
        RememberTool(),
        RevertTurnTool(),
        RlmTool(),
    ]:
        registry.register(tool)

    return registry
