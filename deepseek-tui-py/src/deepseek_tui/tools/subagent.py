"""Sub-agent management — spawn, track, and communicate with child agents.

Port of `crates/tui/src/tools/subagent/` system.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class SubAgentRecord:
    id: str
    name: str
    status: str = "running"  # running | completed | failed | cancelled
    prompt: str = ""
    result: Optional[str] = None
    created_at: str = ""
    process: Optional[asyncio.subprocess.Process] = None


class SubAgentManager:
    """Manages spawned sub-agent processes."""

    def __init__(self) -> None:
        self._agents: dict[str, SubAgentRecord] = {}

    async def spawn(
        self,
        prompt: str,
        name: str = "agent",
        model: Optional[str] = None,
    ) -> SubAgentRecord:
        """Spawn a sub-agent by running the CLI in exec mode."""
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        record = SubAgentRecord(
            id=agent_id,
            name=name,
            status="running",
            prompt=prompt,
            created_at=now,
        )

        # Launch as a subprocess running deepseek exec
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "deepseek_tui.cli", "exec",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            record.process = proc
            self._agents[agent_id] = record

            # Start watching the process
            asyncio.create_task(self._watch_agent(agent_id))
        except Exception as e:
            record.status = "failed"
            record.result = str(e)
            self._agents[agent_id] = record

        return record

    async def _watch_agent(self, agent_id: str) -> None:
        """Watch a sub-agent process and collect results."""
        record = self._agents.get(agent_id)
        if record is None or record.process is None:
            return

        try:
            stdout, stderr = await asyncio.wait_for(
                record.process.communicate(), timeout=300
            )
            record.status = "completed" if record.process.returncode == 0 else "failed"
            output = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            record.result = output
            if err:
                record.result += f"\nSTDERR:\n{err}"
        except asyncio.TimeoutError:
            record.status = "failed"
            record.result = "Timed out after 300s"
            if record.process:
                record.process.kill()
        except Exception as e:
            record.status = "failed"
            record.result = str(e)

    def get(self, agent_id: str) -> Optional[SubAgentRecord]:
        return self._agents.get(agent_id)

    def list(self) -> list[SubAgentRecord]:
        return list(self._agents.values())

    async def cancel(self, agent_id: str) -> bool:
        record = self._agents.get(agent_id)
        if record is None or record.process is None:
            return False
        record.process.kill()
        record.status = "cancelled"
        return True

    async def send_input(self, agent_id: str, input: str) -> bool:
        record = self._agents.get(agent_id)
        if record is None or record.process is None or record.process.stdin is None:
            return False
        try:
            record.process.stdin.write((input + "\n").encode())
            await record.process.stdin.drain()
            return True
        except Exception:
            return False


# Global manager instance
_agent_manager = SubAgentManager()
