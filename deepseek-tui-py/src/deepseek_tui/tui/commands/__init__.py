"""Slash command registry and dispatch system.

Port of `crates/tui/src/commands/mod.rs`.
Provides modular command dispatch, fuzzy matching, and autocomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CommandResult:
    """Result of executing a slash command."""
    message: Optional[str] = None
    action: Optional[str] = None  # Action name for the app to take
    is_error: bool = False

    @classmethod
    def ok(cls) -> CommandResult:
        return cls()

    @classmethod
    def msg(cls, text: str) -> CommandResult:
        return cls(message=text)

    @classmethod
    def error(cls, text: str) -> CommandResult:
        return cls(message=f"Error: {text}", is_error=True)

    @classmethod
    def action(cls, action: str) -> CommandResult:
        return cls(action=action)


CommandHandler = Callable[..., CommandResult]


@dataclass
class CommandInfo:
    """Metadata for help and autocomplete."""
    name: str
    aliases: list[str] = field(default_factory=list)
    usage: str = ""
    description: str = ""
    handler: Optional[CommandHandler] = None

    def requires_argument(self) -> bool:
        return "<" in self.usage or "[" in self.usage

    def palette_command(self) -> str:
        if self.requires_argument():
            return f"/{self.name} "
        return f"/{self.name}"


class CommandRegistry:
    """Registry of all slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, CommandInfo] = {}
        self._alias_map: dict[str, str] = {}

    def register(self, cmd: CommandInfo) -> None:
        name = cmd.name
        self._commands[name] = cmd
        for alias in cmd.aliases:
            self._alias_map[alias] = name

    def get(self, name: str) -> Optional[CommandInfo]:
        name = name.lstrip("/").lower()
        if name in self._commands:
            return self._commands[name]
        if name in self._alias_map:
            primary = self._alias_map[name]
            return self._commands.get(primary)
        # Fuzzy match
        return self._fuzzy_match(name)

    def _fuzzy_match(self, name: str) -> Optional[CommandInfo]:
        """Find best fuzzy match using edit distance."""
        best_dist = 3  # max edit distance
        best_cmd = None
        for cmd_name, cmd in self._commands.items():
            dist = _edit_distance(name, cmd_name)
            if dist < best_dist:
                best_dist = dist
                best_cmd = cmd
            for alias in cmd.aliases:
                dist = _edit_distance(name, alias)
                if dist < best_dist:
                    best_dist = dist
                    best_cmd = cmd
        return best_cmd

    def all(self) -> list[CommandInfo]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    def matching(self, prefix: str) -> list[CommandInfo]:
        prefix = prefix.lstrip("/").lower()
        result = []
        for cmd in self._commands.values():
            if cmd.name.startswith(prefix):
                result.append(cmd)
            elif any(a.startswith(prefix) for a in cmd.aliases):
                result.append(cmd)
        return result

    def names_matching(self, prefix: str) -> list[str]:
        return [c.palette_command() for c in self.matching(prefix)]


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ac in enumerate(a, 1):
        curr[0] = i
        for j, bc in enumerate(b, 1):
            cost = 0 if ac == bc else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[-1]


# Global registry
_registry = CommandRegistry()


def register(cmd: CommandInfo) -> None:
    _registry.register(cmd)


def get_registry() -> CommandRegistry:
    return _registry


def dispatch(input: str, **kwargs) -> CommandResult:
    """Parse and dispatch a slash command."""
    input = input.strip()
    parts = input.split(maxsplit=1)
    cmd_name = parts[0].lstrip("/").lower() if parts[0].startswith("/") else parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    cmd = _registry.get(cmd_name)
    if cmd is None:
        close = _registry.names_matching(cmd_name)
        if close:
            return CommandResult.error(
                f"Unknown command: /{cmd_name}. Did you mean: {', '.join(close[:3])}?"
            )
        return CommandResult.error(
            f"Unknown command: /{cmd_name}. Type /help for available commands."
        )

    if cmd.handler is None:
        return CommandResult.error(f"Command /{cmd.name} has no handler registered.")

    return cmd.handler(arg=arg, **kwargs)
