# -*- coding: utf-8 -*-
"""Text formatting helpers used across the CLI and TUI."""

from __future__ import annotations

import re


def mask_secret(value: str) -> str:
    """Mask a secret showing only first/last few chars with `...` in
    the middle."""
    if not value:
        return ""
    if len(value) > 8:
        return f"{value[:4]}...{value[-4:]}"
    if len(value) >= 2:
        return f"{value[0]}...{value[-1]}"
    return f"{value}..."


def truncate(text: str, max_len: int) -> str:
    """Simple suffix truncation with an ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate long text while preserving head and tail with a marker."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return (
        text[:head]
        + (f"\n... [truncated {len(text) - max_chars} chars] ...\n")
        + text[-tail:]
    )


def truncate_head_tail(text: str, max_chars: int, ratio: float = 0.6) -> str:
    """Truncate long text while preserving beginning and end.

    The head keeps the first (ratio * max_chars) characters; the tail keeps
    the remainder from the end.  A small marker is inserted so the model knows
    content was omitted.
    """
    if len(text) <= max_chars:
        return text
    head_len = int(max_chars * ratio)
    tail_len = max_chars - head_len
    head = text[:head_len]
    tail = text[-tail_len:] if tail_len > 0 else ""
    return (
        f"{head}\n\n... [omitted {len(text) - max_chars} chars]"
        f" ...\n\n{tail}"
    )


_FRONTMATTER_RE = re.compile(
    r"^---\s*\n.*?\n---\s*\n",
    re.DOTALL,
)


def strip_frontmatter(text: str) -> str:
    """Remove YAML/TOML frontmatter from markdown instructions."""
    return _FRONTMATTER_RE.sub("", text, count=1)


# Confirmation panel: truncate long values to keep the prompt readable.
_MAX_DISPLAY_LINES = 20
_MAX_DISPLAY_CHARS = 800


def truncate_value(value: str) -> str:
    """Truncate a value for display in confirmation panels.

    If the string representation exceeds _MAX_DISPLAY_LINES lines or
    _MAX_DISPLAY_CHARS characters, show the head + a summary line.
    """
    s = str(value)
    lines = s.splitlines()
    total_lines = len(lines)
    total_chars = len(s)

    needs_truncate = (
        total_lines > _MAX_DISPLAY_LINES or total_chars > _MAX_DISPLAY_CHARS
    )
    if not needs_truncate:
        return s

    # Keep roughly the first half of the budget
    keep_lines = min(_MAX_DISPLAY_LINES // 2, 10)
    kept = "\n".join(lines[:keep_lines])

    # Build summary
    parts = []
    if total_lines > _MAX_DISPLAY_LINES:
        parts.append(f"{total_lines} lines")
    if total_chars > _MAX_DISPLAY_CHARS:
        parts.append(f"{total_chars} chars")
    summary = f"  ... (omitted, {', '.join(parts)})"

    return kept + "\n" + summary
