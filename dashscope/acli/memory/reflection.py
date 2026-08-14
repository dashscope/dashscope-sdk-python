# -*- coding: utf-8 -*-
"""
Reflection mechanism — detect failure patterns and adjust strategy.
Monitors tool execution outcomes and provides adaptive guidance.
"""

from __future__ import annotations


class ReflectionTracker:
    """Tracks consecutive failures and provides reflection hints."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.consecutive_failures = 0
        self.last_failed_tools: list[str] = []

    def record_success(self) -> None:
        """Record a successful tool execution."""
        self.consecutive_failures = 0
        self.last_failed_tools = []

    def record_failure(self, tool_name: str) -> None:
        """Record a failed tool execution."""
        self.consecutive_failures += 1
        self.last_failed_tools.append(tool_name)

    def record_tool_execution(self, tool_name: str, success: bool) -> None:
        """Record a tool execution outcome."""
        if success:
            self.record_success()
        else:
            self.record_failure(tool_name)

    def needs_reflection(self) -> bool:
        """Check if reflection hints should be injected."""
        return self.consecutive_failures >= self.threshold

    def get_reflection_hint(self) -> str:
        """Generate a reflection hint for system prompt injection."""
        if not self.needs_reflection():
            return ""

        failed_tools_str = ", ".join(set(self.last_failed_tools))
        return (
            f"\n\n## ⚠️ Reflection hint\n"
            f"Detected {self.consecutive_failures} consecutive "
            f"tool failures ({failed_tools_str}).\n"
            f"Suggestions:\n"
            f"1. Check whether the previous approach is flawed\n"
            f"2. Try different tools or parameters\n"
            f"3. Confirm the requirements with the user\n"
            f"4. For complex tasks, re-plan steps via create_plan\n"
        )

    def get_failure_lesson(self) -> str:
        """Generate a lesson string for experience memory."""
        if self.consecutive_failures < self.threshold:
            return ""
        failed_tools_str = ", ".join(set(self.last_failed_tools))
        return (
            f"{self.consecutive_failures} consecutive failures "
            f"({failed_tools_str}); need a new strategy"
        )

    def reset(self) -> None:
        """Reset the tracker for a new turn."""
        self.consecutive_failures = 0
        self.last_failed_tools = []
