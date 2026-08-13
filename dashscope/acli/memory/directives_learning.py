# -*- coding: utf-8 -*-
"""Directives Auto-Learning: Propose rules from user behavior patterns.

Based on the ACE/MCE pattern (Weng 2026): Agent observes recurring user
behaviors and proposes operational rules ("directives") that the user
can accept or reject.

Example: "我发现你总在做 git commit 后立即 git push，要不要加为 rule？"

Architecture:
- Track tool call sequences and user preferences
- Identify recurring patterns (frequency >= threshold)
- Propose directives via /directives propose
- User accepts via /rule add or rejects
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashscope.acli.config import WORKSPACE_DIR


def _patterns_file() -> Path:
    return WORKSPACE_DIR / "memory" / "behavior_patterns.json"


def _empty_patterns() -> dict[str, Any]:
    return {"sequences": [], "preferences": {}, "proposed_directives": []}


# The patterns file is read on every prompt assembly; memoize on mtime.
_patterns_cache: tuple[Path, int, dict[str, Any]] | None = None


def _load_patterns() -> dict[str, Any]:
    global _patterns_cache
    path = _patterns_file()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return _empty_patterns()
    if _patterns_cache and _patterns_cache[:2] == (path, mtime_ns):
        return _patterns_cache[2]
    try:
        patterns = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_patterns()
    _patterns_cache = (path, mtime_ns, patterns)
    return patterns


def _save_patterns(patterns: dict[str, Any]) -> None:
    global _patterns_cache
    path = _patterns_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(patterns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _patterns_cache = None


def record_tool_sequence(tools: list[str]) -> None:
    """Record a sequence of tools used in a successful turn."""
    if len(tools) < 2:
        return

    patterns = _load_patterns()
    patterns["sequences"].append(
        {
            "tools": tools,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Keep only last 100 sequences
    if len(patterns["sequences"]) > 100:
        patterns["sequences"] = patterns["sequences"][-100:]

    _save_patterns(patterns)


def analyze_patterns() -> list[dict[str, Any]]:
    """Analyze recorded patterns and identify recurring behaviors.

    Returns list of proposed directives with:
    - pattern: description of the recurring behavior
    - directive: suggested rule text
    - frequency: how often this pattern occurs
    - confidence: confidence score (0-1)
    """
    patterns = _load_patterns()
    sequences = patterns.get("sequences", [])

    if len(sequences) < 5:
        return []  # Not enough data

    # Count tool sequence patterns
    sequence_counter: Counter = Counter()
    for seq in sequences:
        tools = tuple(seq.get("tools", []))
        if len(tools) >= 2:
            # Look for adjacent pairs
            for i in range(len(tools) - 1):
                pair = (tools[i], tools[i + 1])
                sequence_counter[pair] += 1

    # Find frequent patterns (>= 3 occurrences)
    proposals = []
    for (tool1, tool2), count in sequence_counter.most_common(10):
        if count >= 3:
            confidence = min(count / 10, 1.0)
            directive = _generate_directive(tool1, tool2, count)
            proposals.append(
                {
                    "pattern": f"在执行 {tool1} 后经常执行 {tool2}",
                    "directive": directive,
                    "frequency": count,
                    "confidence": confidence,
                    "tools": [tool1, tool2],
                },
            )

    return proposals


def _generate_directive(tool1: str, tool2: str, count: int) -> str:
    """Generate a directive text from a tool pattern."""
    # Common patterns
    if tool1 == "run_command" and "git commit" in tool2:
        return "在 git commit 后自动执行 git push"
    elif tool1 == "read_file" and tool2 == "write_file":
        return "读取文件后如需修改，直接使用 write_file 而非 run_command"
    elif tool1 == "write_file" and tool2 == "run_command":
        return "修改文件后自动运行测试或构建命令"
    else:
        return f"在执行 {tool1} 后考虑执行 {tool2}（已观察到 {count} 次）"


def propose_directive(directive: str, rationale: str) -> dict[str, Any]:
    """Create a directive proposal."""
    patterns = _load_patterns()

    proposal = {
        "id": f"dir_{int(datetime.now(timezone.utc).timestamp())}",
        "directive": directive,
        "rationale": rationale,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    patterns["proposed_directives"].append(proposal)
    _save_patterns(patterns)

    return proposal


def list_proposed_directives(status: str = "pending") -> list[dict[str, Any]]:
    """List proposed directives."""
    patterns = _load_patterns()
    proposals = patterns.get("proposed_directives", [])

    if status == "all":
        return proposals
    return [p for p in proposals if p.get("status") == status]


def accept_directive(proposal_id: str, config: Any) -> bool:
    """Accept a proposed directive and add it to user_directives."""
    patterns = _load_patterns()
    proposals = patterns.get("proposed_directives", [])

    for p in proposals:
        if p["id"] == proposal_id and p.get("status") == "pending":
            directive_text = p["directive"]

            # Add to config.user_directives
            if hasattr(config, "user_directives"):
                if directive_text not in config.user_directives:
                    config.user_directives.append(directive_text)
                    config.save_workspace()

            p["status"] = "accepted"
            p["accepted_at"] = datetime.now(timezone.utc).isoformat()
            _save_patterns(patterns)
            return True

    return False


def reject_directive(proposal_id: str) -> bool:
    """Reject a proposed directive."""
    patterns = _load_patterns()
    proposals = patterns.get("proposed_directives", [])

    for p in proposals:
        if p["id"] == proposal_id and p.get("status") == "pending":
            p["status"] = "rejected"
            p["rejected_at"] = datetime.now(timezone.utc).isoformat()
            _save_patterns(patterns)
            return True

    return False


def get_directive_proposals_summary() -> str:
    """Get a summary of pending directive proposals for system prompt
    injection."""
    proposals = list_proposed_directives("pending")
    if not proposals:
        return ""

    lines = ["\n\n## Agent 观察到的行为模式（可提议为规则）"]
    for p in proposals[:2]:  # Show max 2
        lines.append(f"- {p['rationale'][:60]}")
    lines.append("\n使用 /directives 查看详情并决定是否添加为规则")

    return "\n".join(lines)
