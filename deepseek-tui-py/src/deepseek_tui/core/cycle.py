"""Cycle Manager / Handoff — checkpoint-restart for long sessions.

Port of `crates/tui/src/cycle_manager.rs` and `handoff.rs`.

Automatically archives conversation cycles and produces structured
carry-forward briefings when the context window approaches capacity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
if TYPE_CHECKING:
    from ..llm import LlmClient
from ..llm.models import MessageRequest, Message, ContentBlock

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_CYCLE_THRESHOLD_TOKENS = 768_000
DEFAULT_BRIEFING_MAX_TOKENS = 3_000
APPROX_CHARS_PER_TOKEN = 4
CYCLE_ARCHIVE_SCHEMA_VERSION = 1

CYCLE_HANDOFF_TEMPLATE = """You are at a cycle boundary. A new context window is about to start.

Write a `<carry_forward>` block (inside XML tags) that captures:

1. **Key decisions made and why** — so the next cycle doesn't second-guess them.
2. **Active constraints discovered** — things that limit available approaches.
3. **Hypotheses being tested** — what you're still unsure about.
4. **Approaches that failed** — so the next cycle doesn't repeat them.
5. **Open questions** — things the user hasn't answered yet.

Rules:
- Stay under {max_tokens} tokens.
- Do NOT recap tool output, file contents, or step-by-step actions.
- Do NOT include the conversation transcript.
- Only output the `<carry_forward>`...`</carry_forward>` block.
"""


# ── Cycle Config ──────────────────────────────────────────────────────

@dataclass
class ModelCycleConfig:
    threshold_tokens: int = DEFAULT_CYCLE_THRESHOLD_TOKENS
    briefing_max_tokens: int = DEFAULT_BRIEFING_MAX_TOKENS


def _default_per_model() -> dict[str, ModelCycleConfig]:
    return {
        "deepseek-v4-pro": ModelCycleConfig(),
        "deepseek-v4-flash": ModelCycleConfig(),
    }


@dataclass
class CycleConfig:
    enabled: bool = True
    threshold_tokens: int = DEFAULT_CYCLE_THRESHOLD_TOKENS
    briefing_max_tokens: int = DEFAULT_BRIEFING_MAX_TOKENS
    per_model: dict[str, ModelCycleConfig] = field(default_factory=_default_per_model)

    def threshold_for(self, model: str) -> int:
        if model in self.per_model:
            return self.per_model[model].threshold_tokens
        return self.threshold_tokens

    def briefing_max_for(self, model: str) -> int:
        if model in self.per_model:
            return self.per_model[model].briefing_max_tokens
        return self.briefing_max_tokens


# ── Threshold Decision ────────────────────────────────────────────────

def should_advance_cycle(
    active_input_tokens: int,
    reserved_headroom_tokens: int,
    model: str,
    cfg: CycleConfig,
    in_flight: bool,
) -> bool:
    """Decide whether a cycle boundary should fire."""
    if not cfg.enabled or in_flight:
        return False
    threshold = cfg.threshold_for(model)
    if threshold == 0:
        return False
    # Consider context window limits
    window = _context_window_for(model)
    if window is not None:
        window_floor = max(0, window - reserved_headroom_tokens)
        threshold = min(threshold, window_floor)
    return active_input_tokens >= threshold


def _context_window_for(model: str) -> Optional[int]:
    """Approximate context window for known models."""
    lower = model.lower()
    if "v4" in lower or lower in ("deepseek-chat", "deepseek-reasoner", "deepseek-r1"):
        return 1_000_000
    if "deepseek" in lower:
        return 128_000
    if "claude" in lower:
        return 200_000
    return None


# ── Briefing ──────────────────────────────────────────────────────────

@dataclass
class CycleBriefing:
    cycle: int
    timestamp: str
    briefing_text: str
    token_estimate: int


async def produce_briefing(
    client: LlmClient,
    model: str,
    conversation: list[dict],
    max_briefing_tokens: int,
    cycle_n: int,
) -> CycleBriefing:
    """Run the briefing turn to produce a <carry_forward> block."""
    if not conversation:
        return CycleBriefing(cycle_n, _now_iso(), "", 0)

    messages = list(conversation)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"[CYCLE BOUNDARY] The next turn starts in a fresh context.\n\n"
                    f"Produce your `<carry_forward>` block now. "
                    f"Stay under {max_briefing_tokens} tokens. "
                    f"Output only the block — no other text."
                ),
            }
        ],
    })

    system_prompt = CYCLE_HANDOFF_TEMPLATE.format(max_tokens=max_briefing_tokens)

    llm_messages = [_msg_to_llm(m) for m in messages]

    req = MessageRequest(
        model=model,
        messages=llm_messages,
        max_tokens=max(max_briefing_tokens * 2, 1024),
        temperature=0.2,
        stream=False,
    )

    try:
        response = await client.create_message(req)
        raw = ""
        for block in response.content:
            t = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if t:
                raw += t
    except Exception as e:
        # If briefing fails, return empty — the cycle can proceed without it
        return CycleBriefing(cycle_n, _now_iso(), f"[Briefing failed: {e}]", 0)

    extracted = extract_carry_forward(raw)
    bounded = enforce_briefing_cap(extracted, max_briefing_tokens)
    return CycleBriefing(
        cycle=cycle_n,
        timestamp=_now_iso(),
        briefing_text=bounded,
        token_estimate=len(bounded) // APPROX_CHARS_PER_TOKEN,
    )


def extract_carry_forward(raw: str) -> str:
    """Extract contents of the first <carry_forward> block."""
    lower = raw.lower()
    open_tag = "<carry_forward>"
    close_tag = "</carry_forward>"

    start = lower.find(open_tag)
    if start == -1:
        return raw.strip()

    after = start + len(open_tag)
    end = lower[after:].find(close_tag)
    if end == -1:
        return raw[after:].strip()

    return raw[after:after + end].strip()


def enforce_briefing_cap(text: str, max_tokens: int) -> str:
    """Defensive bound on briefing length (~4 chars/token)."""
    max_chars = max_tokens * APPROX_CHARS_PER_TOKEN
    if not max_chars:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...briefing truncated to fit cap...]"


# ── Structured State ──────────────────────────────────────────────────

@dataclass
class StructuredState:
    mode_label: str = ""
    workspace: str = ""
    cwd: Optional[str] = None
    working_set_summary: Optional[str] = None
    todo_items: list[dict] = field(default_factory=list)
    plan_steps: list[dict] = field(default_factory=list)
    subagent_snapshots: list[dict] = field(default_factory=list)
    active_tool: Optional[str] = None
    status_line: str = ""

    def to_system_block(self) -> Optional[str]:
        """Render as a system block for the next cycle."""
        out = ["## Cycle State (Auto-Preserved)\n"]
        out.append(f"- Mode: `{self.mode_label}`")
        out.append(f"- Workspace: `{self.workspace}`")
        if self.cwd:
            out.append(f"- Cwd: `{self.cwd}`")

        if self.plan_steps:
            out.append("\n### Plan")
            for s in self.plan_steps:
                status = s.get("status", "pending")
                marker = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(status, "[ ]")
                out.append(f"- {marker} {s.get('step', '')}")

        if self.todo_items:
            done = sum(1 for t in self.todo_items if t.get("status") == "completed")
            pct = int(done / max(len(self.todo_items), 1) * 100)
            out.append(f"\n### Todos ({pct}% complete)")
            for t in self.todo_items:
                status = t.get("status", "pending")
                marker = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(status, "[ ]")
                out.append(f"- {marker} {t.get('content', '')}")

        if self.subagent_snapshots:
            out.append("\n### Open Sub-Agents")
            for s in self.subagent_snapshots:
                out.append(f"- `{s.get('id', '?')}` — {s.get('name', '')}")

        if self.working_set_summary:
            out.append(f"\n{self.working_set_summary}")

        return "\n".join(out) if len(out) > 1 else None

    @classmethod
    def capture(
        cls,
        mode_label: str = "",
        workspace: str = "",
        cwd: Optional[str] = None,
        todo_items: Optional[list[dict]] = None,
        plan_steps: Optional[list[dict]] = None,
        subagent_snapshots: Optional[list[dict]] = None,
        working_set_summary: Optional[str] = None,
    ) -> StructuredState:
        return cls(
            mode_label=mode_label,
            workspace=workspace,
            cwd=cwd,
            todo_items=todo_items or [],
            plan_steps=plan_steps or [],
            subagent_snapshots=subagent_snapshots or [],
            working_set_summary=working_set_summary,
        )


# ── Seed Builder ──────────────────────────────────────────────────────

def build_seed_messages(
    structured_state_block: Optional[str],
    briefing: Optional[CycleBriefing],
    pending_user_message: Optional[str],
) -> list[dict]:
    """Compose the seed messages for the next cycle.

    Returns a list of message dicts:
    1. Structured state user message + assistant ack
    2. Briefing user message + assistant ack
    3. Pending user message (if any)
    """
    out: list[dict] = []

    if structured_state_block and structured_state_block.strip():
        out.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "[CYCLE STATE — auto-preserved across the cycle boundary]\n\n"
                        f"{structured_state_block.strip()}"
                    ),
                }
            ],
        })
        out.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "Acknowledged. State carried into the new cycle."}],
        })

    if briefing and briefing.briefing_text.strip():
        out.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"[CYCLE BRIEFING — written by you on cycle {briefing.cycle} "
                        f"at {briefing.timestamp}]\n\n"
                        f"<carry_forward>\n{briefing.briefing_text.strip()}\n</carry_forward>"
                    ),
                }
            ],
        })
        out.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "Briefing absorbed. Continuing."}],
        })

    if pending_user_message and pending_user_message.strip():
        out.append({
            "role": "user",
            "content": [{"type": "text", "text": pending_user_message.strip()}],
        })

    return out


# ── Archive ───────────────────────────────────────────────────────────

def archive_cycle(
    session_id: str,
    cycle_n: int,
    messages: list[dict],
    model: str,
    archive_dir: Optional[Path] = None,
) -> Path:
    """Archive a cycle's messages to JSONL on disk."""
    dir = archive_dir or _default_archive_dir_for(session_id)
    dir.mkdir(parents=True, exist_ok=True)

    path = dir / f"{cycle_n}.jsonl"
    header = {
        "schema_version": CYCLE_ARCHIVE_SCHEMA_VERSION,
        "cycle": cycle_n,
        "session_id": session_id,
        "model": model,
        "started": _now_iso(),
        "ended": _now_iso(),
        "message_count": len(messages),
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    return path


def open_archive(path: Path) -> tuple[dict, list[dict]]:
    """Open an archived cycle JSONL and return (header, messages)."""
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        header = json.loads(header_line)

        if header.get("schema_version", 0) > CYCLE_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                f"Archive schema v{header['schema_version']} is newer than supported "
                f"v{CYCLE_ARCHIVE_SCHEMA_VERSION}"
            )

        messages = []
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    return header, messages


def list_cycles(session_id: str, archive_dir: Optional[Path] = None) -> list[int]:
    """List cycle numbers available for a session, sorted ascending."""
    dir = archive_dir or _default_archive_dir_for(session_id)
    if not dir.exists():
        return []
    cycles = []
    for f in sorted(dir.glob("*.jsonl")):
        try:
            num = int(f.stem)
            cycles.append(num)
        except ValueError:
            continue
    return cycles


# ── Handoff Thresholds ───────────────────────────────────────────────

THRESHOLDS: list[tuple[float, str]] = [
    (0.9, "Context at 90%: stop and write handoff now"),
    (0.8, "Context at 80%: draft handoff"),
    (0.7, "Context at 70%: consider wrapping current sub-task"),
]


def threshold_message(ratio: float) -> Optional[str]:
    """Return a handoff message if the context ratio exceeds a threshold."""
    for t, msg in THRESHOLDS:
        if ratio >= t:
            return msg
    return None


# ── Helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_archive_dir_for(session_id: str) -> Path:
    return Path.home() / ".deepseek" / "sessions" / session_id / "cycles"


def _msg_to_llm(msg: dict) -> Message:
    """Convert a dict message to an LLM Message object."""
    return Message(role=msg.get("role", "user"), content=[
        ContentBlock(type=b.get("type", "text"), text=b.get("text", ""))
        for b in msg.get("content", [{"type": "text", "text": ""}])
    ])


# ── Recall Archive Tool ───────────────────────────────────────────────

class RecallArchiveTool:
    """Search past cycle archives for matching content.

    Port of the planned `recall_archive` tool from issue #127.
    """

    name = "recall_archive"
    description = "Search past cycle archives for decisions, constraints, or reasoning."

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for in archived cycle briefings and messages.",
            },
            "session_id": {
                "type": "string",
                "description": "Session ID to search (defaults to current).",
            },
            "max_cycles": {
                "type": "integer",
                "description": "Maximum number of recent cycles to search.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, session_id: Optional[str] = None) -> None:
        self._session_id = session_id

    async def execute(self, input: dict) -> dict:
        query = input.get("query", "").lower()
        session_id = input.get("session_id") or self._session_id
        max_cycles = input.get("max_cycles", 5)

        if not query:
            return {"ok": False, "error": "Missing required field 'query'"}

        if not session_id:
            return {"ok": False, "error": "No session ID available"}

        cycles = list_cycles(session_id)
        if not cycles:
            return {"ok": True, "results": [], "message": "No archived cycles found."}

        results = []
        for cycle_n in cycles[-max_cycles:]:
            path = _default_archive_dir_for(session_id) / f"{cycle_n}.jsonl"
            try:
                header, messages = open_archive(path)
                matched_messages = []
                for msg in messages:
                    content = json.dumps(msg)
                    if query in content.lower():
                        preview = (msg.get("content", [{}])[0] if msg.get("content") else {}).get("text", "")
                        matched_messages.append({
                            "role": msg.get("role", ""),
                            "preview": preview[:200],
                        })
                if matched_messages:
                    results.append({
                        "cycle": cycle_n,
                        "model": header.get("model", "?"),
                        "matches": len(matched_messages),
                        "messages": matched_messages[:5],  # Limit per cycle
                    })
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        return {
            "ok": True,
            "results": results,
            "total": sum(r["matches"] for r in results),
        }
