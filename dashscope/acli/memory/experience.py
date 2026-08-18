# -*- coding: utf-8 -*-
"""
Experience memory — learn from task outcomes to improve future performance.
Records tool usage patterns, success/failure outcomes, and lessons learned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dashscope.acli.utils.keywords import (
    expand_scoring_terms,
    extract_keywords,
)


class ExperienceTracker:
    """Tracks and retrieves task experiences for learning."""

    # Rotation bounds: once the file outgrows MAX_FILE_BYTES, keep only the
    # newest KEEP_LINES entries (atomic rewrite).
    MAX_FILE_BYTES = 256 * 1024
    KEEP_LINES = 500

    def __init__(self, workspace_dir: Path):
        self.experience_file = workspace_dir / "experiences.jsonl"
        self._cache: list[dict[str, Any]] | None = None
        self._cache_mtime_ns: int | None = None

    def record_experience(
        self,
        task_summary: str,
        tools_used: list[str],
        outcome: str,  # "success", "failure", "partial"
        lesson: str = "",
    ) -> None:
        """Record a task experience for future reference."""
        from dashscope.acli.utils import sanitize_text

        entry = {
            "task": sanitize_text(task_summary),
            "tools": tools_used,
            "outcome": outcome,
            "lesson": sanitize_text(lesson),
        }
        try:
            with open(self.experience_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            return
        self._rotate_if_oversize()

    def _rotate_if_oversize(self) -> None:
        try:
            if self.experience_file.stat().st_size <= self.MAX_FILE_BYTES:
                return
            lines = self.experience_file.read_text(
                encoding="utf-8",
            ).splitlines()
            keep = [line for line in lines if line.strip()][-self.KEEP_LINES :]
            tmp = self.experience_file.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
            tmp.replace(self.experience_file)
        except OSError:
            pass

    def _load(self) -> list[dict[str, Any]]:
        """Read all experiences, memoized on the file's mtime."""
        try:
            mtime_ns = self.experience_file.stat().st_mtime_ns
        except OSError:
            return []
        if self._cache is not None and self._cache_mtime_ns == mtime_ns:
            return self._cache
        experiences: list[dict[str, Any]] = []
        try:
            with open(self.experience_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        experiences.append(json.loads(line))
        except Exception:
            return []
        self._cache = experiences
        self._cache_mtime_ns = mtime_ns
        return experiences

    def search_experiences(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Search for relevant experiences based on CJK-aware keyword
        overlap."""
        keywords = extract_keywords(query)
        if not keywords:
            return []
        # Single CJK chars match almost everything; expand_scoring_terms
        # prefers multi-char tokens + bigrams and only falls back to raw
        # keywords (single chars) when nothing more specific exists.
        scoring = expand_scoring_terms(keywords)

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, exp in enumerate(self._load()):
            task = exp.get("task", "").lower()
            tools_str = " ".join(exp.get("tools", [])).lower()
            score = 0
            for kw in scoring:
                if kw in task:
                    score += 2
                if kw in tools_str:
                    score += 1
            if score > 0:
                if exp.get("lesson"):
                    score += 1
                scored.append((score, index, exp))

        # Highest score first; ties favor the most recently recorded entry.
        scored.sort(key=lambda item: (-item[0], -item[1]))

        # Dedupe near-identical records, keeping the most recent.
        seen: set[tuple[str, str]] = set()
        results: list[dict[str, Any]] = []
        for _, _, exp in scored:
            key = (exp.get("task", ""), exp.get("lesson", ""))
            if key in seen:
                continue
            seen.add(key)
            results.append(exp)
            if len(results) >= limit:
                break
        return results

    def format_experiences_for_prompt(
        self,
        experiences: list[dict[str, Any]],
    ) -> str:
        """Format experiences for injection into system prompt."""
        if not experiences:
            return ""

        lines = ["\n\n## Past Lessons (reference)"]
        for i, exp in enumerate(experiences, 1):
            task = exp.get("task", "")
            outcome = exp.get("outcome", "unknown")
            lesson = exp.get("lesson", "")
            tools = exp.get("tools", [])

            outcome_emoji = {
                "success": "✓",
                "failure": "✗",
                "partial": "△",
            }.get(
                outcome,
                "?",
            )
            lines.append(f"{i}. {outcome_emoji} {task}")
            if tools:
                lines.append(f"   Tools used: {', '.join(tools)}")
            if lesson:
                lines.append(f"   Lesson: {lesson}")

        return "\n".join(lines)
