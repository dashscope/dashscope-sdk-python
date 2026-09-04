# -*- coding: utf-8 -*-
"""
Reflection mechanism — detect failure patterns and adjust strategy.
Monitors tool execution outcomes and provides adaptive guidance.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

# Tools that never change local state.
_READONLY_TOOLS = frozenset(
    {
        "read_file",
        "search_files",
        "list_directory",
        "memory_search",
        "web_search",
        "image_search",
    },
)

# Shell verbs whose output is purely informational.
_READONLY_VERBS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "find",
        "ls",
        "wc",
        "file",
        "stat",
        "du",
        "df",
        "which",
        "command",
        "echo",
        "printf",
        "pwd",
        "date",
        "uname",
        "env",
        "printenv",
        "diff",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "basename",
        "dirname",
        "id",
        "hostname",
        "sw_vers",
        "ps",
        "pgrep",
        "jq",
        "sort",
        "uniq",
        "tr",
        "cut",
        "column",
        "true",
        "test",
        "free",
        "uptime",
        "nproc",
        "lsblk",
        "dig",
        "nslookup",
        "host",
        "whoami",
        "realpath",
        "type",
    },
)

# Anything containing these marks a mutating command. Deliberately
# conservative: a false positive only suppresses the stagnation nudge.
# Container CLIs (docker/podman/colima) are classified by subcommand below.
_WRITE_MARKERS = re.compile(
    r"(?:^|[\s;|&(])"
    r"(?:rm|mv|cp|mkdir|rmdir|touch|chmod|chown|ln|dd|truncate|"
    r"pip|pip3|uv|npm|npx|pnpm|brew|apt|apt-get|yum|dnf|pacman|"
    r"curl|wget|git|kill|killall|"
    r"make|cargo|go|gradle|mvn|python|python3|node|setsid|nohup|tmux|"
    r"tar|zip|unzip|gzip|sed|awk|patch|tee)\b",
)

# Read-only subcommands for container runtimes.
_CONTAINER_READONLY_SUBCMDS = {
    "docker": {"ps", "images", "logs", "inspect", "stats", "version", "info"},
    "podman": {"ps", "images", "logs", "inspect", "stats", "version", "info"},
    "colima": {"status", "list", "version"},
}

# Diagnostic redirections that never change local state.
_DEVNULL_REDIRECT = re.compile(r"(\d?>>?|&>)\s*/dev/null|2>&1")

_REDIRECT = re.compile(r"[^>&]\s*>[^&]|^\s*>|>>")


def _readonly_shell_segment(segment: str) -> bool:
    """True if every stage of one ';'/&&-free segment is a read verb."""
    stages = segment.split("|")
    for stage in stages:
        stage = stage.strip()
        if not stage:
            continue
        try:
            tokens = shlex.split(stage)
        except ValueError:
            return False
        tokens = [t for t in tokens if not t.startswith("-")]
        if not tokens:
            return False
        verb = tokens[0]
        if verb == "env":
            tokens = [t for t in tokens[1:] if "=" not in t]
            if not tokens:
                continue
            verb = tokens[0]
        if verb in _READONLY_VERBS:
            continue
        if verb in _CONTAINER_READONLY_SUBCMDS:
            subcmds = _CONTAINER_READONLY_SUBCMDS[verb]
            if len(tokens) > 1 and tokens[1] in subcmds:
                continue
            return False
        return False
    return True


def is_readonly_tool_call(  # pylint: disable=too-many-return-statements
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> bool:
    """Classify a tool call as read-only (no local state change).

    Conservative: anything ambiguous is treated as mutating so the
    stagnation nudge never fires on genuinely productive work.
    """
    if tool_name in _READONLY_TOOLS:
        return True
    if tool_name.startswith("mcp_"):
        return False
    if tool_name != "run_command":
        return False
    command = (arguments or {}).get("command", "")
    if not command or not isinstance(command, str):
        return False
    # Strip benign diagnostic redirections before scanning for writes.
    command = _DEVNULL_REDIRECT.sub(" ", command)
    if _REDIRECT.search(command) or "tee " in command:
        return False
    if _WRITE_MARKERS.search(command):
        return False
    segments = re.split(r";|&&|\|\|", command)
    return all(_readonly_shell_segment(seg) for seg in segments if seg.strip())


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


class StagnationTracker:
    """Detects read-only stalls: long runs of inspection calls with no
    action that changes state.

    ReflectionTracker only fires on *failures*; a loop of successful
    grep/cat/check calls resets it every time. This tracker closes that
    blind spot by counting consecutive read-only calls.
    """

    def __init__(self, threshold: int = 8):
        self.threshold = threshold
        self.readonly_streak = 0

    def record(self, readonly: bool) -> None:
        if readonly:
            self.readonly_streak += 1
        else:
            self.readonly_streak = 0

    def needs_nudge(self) -> bool:
        return self.readonly_streak >= self.threshold

    def get_stagnation_hint(self, hard_cap: int | None = None) -> str:
        """Imperative convergence nudge; escalates as the streak grows."""
        if not self.needs_nudge():
            return ""
        n = self.readonly_streak
        lines = [
            "\n\n## 🛑 Stagnation warning",
            (
                f"{n} consecutive read-only tool calls with no change "
                "executed (no writes, no builds, no commands with side "
                "effects). You are verifying, not progressing."
            ),
            "Required, in order:",
            "1. STOP gathering information — you already have enough.",
            "2. Execute the FIRST concrete action NOW "
            "(write/edit/run the actual change).",
            (
                "3. If you are genuinely blocked, ask the user one "
                "specific question instead of checking again."
            ),
            "Do NOT announce an action and then inspect more instead.",
        ]
        if hard_cap and hard_cap > n:
            lines.append(
                f"Hard stop in {hard_cap - n} more read-only calls: "
                "produce your best-effort result and finish.",
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self.readonly_streak = 0


def convergence_hint(
    loop_index: int,
    max_turns: int,
    soft_ratio: float = 0.6,
    hard_ratio: float = 0.85,
) -> str:
    """Budget-aware converge/switch nudge for autonomous (oneshot) runs.

    Third non-convergence detector. ReflectionTracker fires on *failures*,
    StagnationTracker on *read-only stalls*; this covers the case where the
    agent does productive, successful, mutating work that nonetheless
    plateaus and burns the whole turn budget without reaching the goal
    (e.g. a renderer stuck just under a similarity threshold). Keyed on the
    fraction of the turn budget consumed, so it needs no task metric.

    Returns "" below ``soft_ratio``; a switch-approach-or-lock-in nudge in
    the soft band; a finalize-now nudge at/after ``hard_ratio``. A
    ``soft_ratio >= 1.0`` disables it; ``max_turns <= 0`` is a safe no-op.
    """
    if max_turns <= 0 or soft_ratio >= 1.0:
        return ""
    used = loop_index + 1
    frac = used / max_turns
    remaining = max(0, max_turns - used)
    if frac >= hard_ratio:
        return (
            "\n\n## ⏳ Budget nearly exhausted — converge now\n"
            f"You have used {used}/{max_turns} iterations "
            f"({remaining} left).\n"
            "Stop refining. With your remaining turns:\n"
            "1. Keep the best version you have produced so far.\n"
            "2. Run ONE final verification against the acceptance "
            "criterion.\n"
            "3. Write the final result and finish.\n"
            "Do NOT start a new approach now — there is no budget left to "
            "debug it."
        )
    if frac >= soft_ratio:
        return (
            "\n\n## ⏳ Budget check — converge or switch approach\n"
            f"You have used {used}/{max_turns} iterations "
            f"({remaining} left).\n"
            "Assess progress honestly: is your result measurably closer to "
            "the goal than it was several iterations ago?\n"
            "- If NO (plateaued): this approach is not working. Switch to a "
            "structurally different strategy now instead of tweaking the "
            "same one.\n"
            "- If YES (improving): continue, but reserve the last ~15% of "
            "your budget to finalize and self-verify.\n"
            "- If the criterion is ALREADY met: stop optimizing, do one "
            "final self-verify, and finish — do not risk regressing a "
            "passing result."
        )
    return ""
