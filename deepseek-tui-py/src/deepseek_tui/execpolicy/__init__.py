"""Exec policy engine — port of `deepseek-execpolicy` crate.

Evaluates tool execution requests against rules and decides approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config.model import RunMode


class ExecApprovalRequirementKind(Enum):
    SKIP = "skip"
    NEEDS_APPROVAL = "needs_approval"
    FORBIDDEN = "forbidden"


@dataclass
class ExecApprovalRequirement:
    kind: ExecApprovalRequirementKind
    reason: str = ""
    # For SKIP
    bypass_sandbox: bool = False
    # For NEEDS_APPROVAL
    proposed_prefixes: list[str] = field(default_factory=list)

    @classmethod
    def skip(cls, reason: str = "", bypass_sandbox: bool = False) -> ExecApprovalRequirement:
        return cls(ExecApprovalRequirementKind.SKIP, reason, bypass_sandbox)

    @classmethod
    def needs_approval(cls, reason: str = "", prefixes: Optional[list[str]] = None) -> ExecApprovalRequirement:
        return cls(ExecApprovalRequirementKind.NEEDS_APPROVAL, reason, proposed_prefixes=prefixes or [])

    @classmethod
    def forbidden(cls, reason: str = "") -> ExecApprovalRequirement:
        return cls(ExecApprovalRequirementKind.FORBIDDEN, reason)


@dataclass
class ExecPolicyContext:
    command: str
    cwd: str
    tool: str = ""
    ask_for_approval: bool = True
    sandbox_mode: Optional[str] = None


@dataclass
class ExecPolicyDecision:
    allow: bool
    requires_approval: bool
    requirement: ExecApprovalRequirement
    matched_rule: Optional[str] = None

    def reason(self) -> str:
        return self.requirement.reason


class ExecPolicyEngine:
    """Simple policy engine for tool execution approval."""
    READ_ONLY_TOOLS = {"read_file", "grep", "glob", "list_dir", "search", "web_fetch"}

    def __init__(self, allowed_prefixes: Optional[list[str]] = None,
                 denied_prefixes: Optional[list[str]] = None) -> None:
        self._allowed = allowed_prefixes or []
        self._denied = denied_prefixes or ["rm -rf", "sudo", "> /dev/sda", "mkfs", "dd if="]

    def check(self, context: ExecPolicyContext, mode: RunMode = RunMode.AGENT) -> ExecPolicyDecision:
        # YOLO mode: always skip (auto-approve)
        if mode == RunMode.YOLO:
            return ExecPolicyDecision(
                allow=True,
                requires_approval=False,
                requirement=ExecApprovalRequirement.skip("YOLO mode - auto-approved"),
            )

        # Plan mode: only allow read-only tools
        if mode == RunMode.PLAN:
            tool = context.tool
            if tool in self.READ_ONLY_TOOLS:
                return ExecPolicyDecision(
                    allow=True,
                    requires_approval=False,
                    requirement=ExecApprovalRequirement.skip("Plan mode read-only"),
                )
            return ExecPolicyDecision(
                allow=False,
                requires_approval=False,
                requirement=ExecApprovalRequirement.forbidden(
                    f"Plan mode: write tool '{tool}' is not allowed"
                ),
            )

        # Agent mode: use normal approval rules (original logic)
        command = context.command.strip()

        # Check denied prefixes
        for prefix in self._denied:
            if command.startswith(prefix):
                return ExecPolicyDecision(
                    allow=False,
                    requires_approval=False,
                    requirement=ExecApprovalRequirement.forbidden(
                        f"Command matches denied prefix: '{prefix}'"
                    ),
                    matched_rule=f"deny:{prefix}",
                )

        # Check allowed prefixes
        for prefix in self._allowed:
            if command.startswith(prefix):
                return ExecPolicyDecision(
                    allow=True,
                    requires_approval=False,
                    requirement=ExecApprovalRequirement.skip("Matched allowed prefix"),
                    matched_rule=f"allow:{prefix}",
                )

        # Default: require approval if ask_for_approval is True
        if context.ask_for_approval:
            return ExecPolicyDecision(
                allow=True,
                requires_approval=True,
                requirement=ExecApprovalRequirement.needs_approval(
                    f"Command '{command}' requires approval",
                ),
            )

        return ExecPolicyDecision(
            allow=True,
            requires_approval=False,
            requirement=ExecApprovalRequirement.skip("Approval not requested"),
        )
