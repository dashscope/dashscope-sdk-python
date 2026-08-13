# -*- coding: utf-8 -*-
"""
Two-tier Memory Architecture for Agent Context Management.

Working context: the conversation message list, owned by the agent loop and
  compressed by acli.compression (not tracked in this module).
Mid-term (Session Memory): Per-session persistent state
  (plan, reflection, tool_chains)
Long-term (Persistent Memory): Cross-session persistent memory
  (experience, user profile)

Design inspired by context engineering principles (Weng 2026,
TMLR Survey 2026).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dashscope.acli.memory.experience import ExperienceTracker
from dashscope.acli.memory.plan import PlanTracker
from dashscope.acli.memory.reflection import ReflectionTracker
from dashscope.acli.memory.tool_chains import ToolChainLibrary
from dashscope.acli.memory.trace import TraceLogger


class SessionMemory:
    """Mid-term memory: per-session persistent state.

    Tracks plan, reflection, and tool chains for the current session.
    Persisted to .acli/session/ directory.
    """

    def __init__(self, workspace_dir: Path):
        self.session_dir = workspace_dir / "session"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.plan = PlanTracker()
        self.reflection = ReflectionTracker()
        self.tool_chains = ToolChainLibrary(self.session_dir)

    def reset(self) -> None:
        """Reset session memory (called at session start)."""
        self.plan.clear_plan()
        self.reflection.reset()


class PersistentMemory:
    """Long-term memory: cross-session persistent memory.

    Stores experience and user profile that persist across sessions.
    Persisted to .acli/ directory.
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.experience = ExperienceTracker(workspace_dir)

    def record_experience(
        self,
        task_summary: str,
        tools_used: list[str],
        outcome: str,
        lesson: str = "",
    ) -> None:
        """Record a task experience."""
        self.experience.record_experience(
            task_summary,
            tools_used,
            outcome,
            lesson,
        )

    def search_experiences(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Search for relevant experiences."""
        return self.experience.search_experiences(query, limit)

    def format_for_prompt(self, experiences: list[dict[str, Any]]) -> str:
        """Format experiences for prompt injection."""
        return self.experience.format_experiences_for_prompt(experiences)


class MemoryManager:
    """Two-tier memory manager: per-session state + cross-session
    persistent memory.

    Coordinates mid-term (session: plan, reflection, tool chains) and
    long-term (persistent: experience, user profile) memory, plus trace
    logging. The conversation's working context lives in Agent.messages and
    is compressed by acli.compression — not tracked here.
    """

    def __init__(self, workspace_dir: Path):
        self.session = SessionMemory(workspace_dir)
        self.persistent = PersistentMemory(workspace_dir)
        self.trace = TraceLogger(workspace_dir)

    @classmethod
    def derive_child(cls, parent: "MemoryManager") -> "MemoryManager":
        """Build an isolated manager for a sub-agent.

        The child gets a fresh SessionMemory (own plan/reflection/tool_chains,
        under a ``child-*`` subdirectory) so its tool failures and plans never
        touch the parent's counters, while sharing the parent's persistent
        experience store and trace logger.
        """
        import uuid

        child = cls.__new__(cls)
        child_workspace = (
            parent.session.session_dir / "children" / uuid.uuid4().hex[:8]
        )
        child.session = SessionMemory(child_workspace)
        child.persistent = parent.persistent
        child.trace = parent.trace
        return child

    def record_tool_execution(self, tool_name: str, success: bool) -> None:
        """Record a tool execution for reflection monitoring."""
        self.session.reflection.record_tool_execution(tool_name, success)

    def record_experience(
        self,
        task_summary: str,
        tools_used: list[str],
        outcome: str,
        lesson: str = "",
    ) -> None:
        """Record a task experience to persistent memory."""
        self.persistent.record_experience(
            task_summary,
            tools_used,
            outcome,
            lesson,
        )

    def log_trace(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a trace event."""
        self.trace.log(event_type, data)

    def reset_session(self) -> None:
        """Reset session memory (called at session start)."""
        self.session.reset()
