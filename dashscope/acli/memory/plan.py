# -*- coding: utf-8 -*-
"""
Planning layer for complex tasks.
Provides plan creation, tracking, and progress reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PlanStep:
    """A single step in a plan."""

    description: str
    completed: bool = False
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """A multi-step execution plan."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def progress_summary(self) -> str:
        """Generate a progress summary for system prompt injection."""
        if not self.steps:
            return ""
        completed = sum(1 for s in self.steps if s.completed)
        total = len(self.steps)
        lines = [f"\nPlan progress: {completed}/{total}"]
        for i, step in enumerate(self.steps, 1):
            status = "✓" if step.completed else "→"
            lines.append(f"  {status} {i}. {step.description}")
        return "\n".join(lines)

    def mark_step_complete(self, step_index: int, tool_name: str = "") -> None:
        """Mark a step as completed."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].completed = True
            if tool_name:
                self.steps[step_index].tool_calls.append(tool_name)


class PlanTracker:
    """Tracks the current plan across tool calls.

    Plans expire: once a plan is fully completed or outlives ``ttl_seconds``
    it is dropped and no longer injected into the prompt. This prevents a
    stale, abandoned plan from steering every later turn of the session.
    """

    def __init__(
        self,
        ttl_seconds: float = 1800,
        clock: Callable[[], float] | None = None,
    ):
        self.current_plan: Plan | None = None
        self.ttl_seconds = ttl_seconds
        self._clock = clock or time.monotonic
        self._created_at: float | None = None

    def create_plan(self, goal: str, steps: list[str]) -> str:
        """Create a new plan. Returns confirmation message."""
        self.current_plan = Plan(
            goal=goal,
            steps=[PlanStep(description=desc) for desc in steps],
        )
        self._created_at = self._clock()
        return f"Plan created: {goal}\nSteps: {len(steps)}"

    def _expire_if_stale(self) -> None:
        if self.current_plan is None:
            return
        if self.is_complete():
            self.clear_plan()
            return
        if (
            self._created_at is not None
            and self._clock() - self._created_at > self.ttl_seconds
        ):
            self.clear_plan()

    def get_plan_section(self) -> str:
        """Get plan progress for system prompt injection."""
        self._expire_if_stale()
        if not self.current_plan:
            return ""
        return self.current_plan.progress_summary()

    def mark_step_complete(self, step_index: int, tool_name: str = "") -> None:
        """Mark a plan step as completed."""
        self._expire_if_stale()
        if self.current_plan:
            self.current_plan.mark_step_complete(step_index, tool_name)

    def clear_plan(self) -> None:
        """Clear the current plan."""
        self.current_plan = None
        self._created_at = None

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        if not self.current_plan:
            return False
        return all(s.completed for s in self.current_plan.steps)
