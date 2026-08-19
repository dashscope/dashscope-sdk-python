# -*- coding: utf-8 -*-
"""Conversation history memory — auto-store summaries after each turn.

Storage layout:
    .acli/memory/history.json

Each entry:
    {
        "id": "uuid",
        "summary": "...",
        "created_at": "ISO timestamp",
        "turns": 5
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dashscope.acli.config import WORKSPACE_DIR
from dashscope.acli.utils import is_tool_garbage, sanitize_text
from dashscope.acli.utils.ids import now_iso, short_uuid
from dashscope.acli.utils.keywords import extract_keywords
from dashscope.acli.utils.paths import atomic_write_text

# Patterns that pollute history summaries but carry no semantic meaning.
_NOISE_PREFIXES = [
    r"^任务\s*[\d一二三四五六七八九十]+\s*[:：]\s*",
    r"^任务\s*[\d一二三四五六七八九十]+\s+",
    r"^task\s*\d+\s*[:：]\s*",
    r"^task\s*\d+\s+",
    r"^[\(\[（]\s*\d+\s*[\)\]）]\s*[:：]?\s*",
    r"^\d+[\.、\)）]\s*",
]
_FOLLOWUP_ONLY = {
    "继续",
    "继续。",
    "ok",
    "okay",
    "好的",
    "好",
    "好。",
    "yes",
    "y",
    "嗯",
    "嗯。",
    "行",
    "行。",
    # English equivalents of the Chinese follow-ups above
    "continue",
    "continue.",
    "yeah",
    "yeah.",
    "fine",
    "fine.",
    "sure",
    "sure.",
}


def _clean_summary_text(text: str) -> str:
    """Strip task-number prefixes and other noise from a candidate summary."""
    cleaned = text.strip()
    for pattern in _NOISE_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.lower() in _FOLLOWUP_ONLY:
        return ""
    return cleaned


def _history_file():
    return WORKSPACE_DIR / "memory" / "history.json"


def _load() -> list[dict[str, Any]]:
    path = _history_file()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    path = _history_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(entries, ensure_ascii=False, indent=2))


def extract_summary(messages: list[dict[str, str]]) -> str:
    """Extract a brief summary from conversation messages.

    Strategy: first meaningful user message + key tool calls.
    Skips compressed/auto-generated user messages that contain raw tool-call
    scaffolding, because those pollute memory recall. Also strips common
    noise prefixes like ``Task 1:``/``1. `` (and their Chinese
    equivalents) and ignores bare follow-ups (``ok``, ``continue``, and
    their Chinese equivalents) so the summary reflects the actual intent
    rather than turn markers.
    """
    if not messages:
        return ""

    # Find the first user message that is not tool garbage or follow-up noise.
    candidate = ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        text = content[:200].strip()
        if not text or is_tool_garbage(text):
            continue
        cleaned = _clean_summary_text(text[:100])
        if len(cleaned) >= 5:
            candidate = cleaned
            break

    if not candidate:
        return ""

    # Key tool calls
    tools_used: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                if name and name not in tools_used:
                    tools_used.append(name)

    if tools_used:
        return sanitize_text(
            f"{candidate} [tools: {', '.join(tools_used[:3])}]",
        )
    return sanitize_text(candidate)


def store_history(messages: list[dict[str, str]]) -> dict[str, Any] | None:
    """Store a conversation summary to history."""
    summary = extract_summary(messages)
    if not summary:
        return None

    turns = sum(1 for m in messages if m.get("role") == "user")
    if turns < 1:
        return None

    entry = {
        "id": short_uuid(),
        "summary": summary,
        "created_at": now_iso(),
        "turns": turns,
    }

    entries = _load()
    entries.append(entry)
    _save(entries)
    return entry


def list_history(limit: int = 20) -> list[dict[str, Any]]:
    """List recent history entries."""
    if limit <= 0:
        return []
    entries = _load()
    return entries[-limit:]


def search_history(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search history by keyword."""
    entries = _load()
    if not entries or not query.strip():
        return []

    keywords = extract_keywords(query)
    if not keywords:
        return []

    scored: list[tuple[float, dict]] = []
    for entry in entries:
        summary_lower = entry.get("summary", "").lower()
        hits = sum(1 for kw in keywords if kw in summary_lower)
        if hits > 0:
            score = min(hits / max(len(keywords), 1), 1.0)
            scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:limit]]


def delete_history(entry_id: str) -> bool:
    """Delete a specific history entry."""
    entries = _load()
    original_len = len(entries)
    entries = [e for e in entries if e["id"] != entry_id]
    if len(entries) < original_len:
        _save(entries)
        return True
    return False


def clear_history() -> int:
    """Clear all history entries."""
    entries = _load()
    count = len(entries)
    if count > 0:
        _save([])
    return count


def stats() -> dict[str, Any]:
    """Return history statistics."""
    entries = _load()
    total_turns = sum(e.get("turns", 0) for e in entries)
    return {
        "count": len(entries),
        "total_turns": total_turns,
    }


def export_history(path: str, fmt: str | None = None) -> str:
    """Export conversation history to a file.

    Supported formats: html (default), json, markdown.
    When ``fmt`` is omitted, the format is auto-detected from the file
    extension (``.json`` -> json, ``.md`` -> markdown, otherwise html).
    Returns the resolved output path.
    """
    entries = _load()
    output = Path(path)

    if fmt is None:
        ext = output.suffix.lower()
        if ext == ".json":
            fmt = "json"
        elif ext == ".md":
            fmt = "markdown"
        else:
            fmt = "html"

    if fmt == "json":
        output.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif fmt == "markdown":
        lines = ["# Conversation History\n"]
        for e in entries:
            lines.append(f"## {e.get('summary', '(no summary)')}\n")
            lines.append(f"- **Time**: {e.get('created_at', '')}")
            lines.append(f"- **Turns**: {e.get('turns', 0)}\n")
        output.write_text("\n".join(lines), encoding="utf-8")
    else:
        output.write_text(_generate_html(entries), encoding="utf-8")

    return str(output)


def _generate_html(entries: list[dict[str, Any]]) -> str:
    """Build self-contained HTML from history entries."""
    rows = []
    for e in entries:
        summary = e.get("summary", "(no summary)")
        created = e.get("created_at", "")
        turns = e.get("turns", 0)
        rows.append(
            f"<tr><td>{_esc(summary)}</td>"
            f"<td>{_esc(created)}</td>"
            f"<td style='text-align:center'>{turns}</td></tr>",
        )

    total_turns = sum(e.get("turns", 0) for e in entries)
    total_count = len(entries)

    return (
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgenticCLI Conversation History</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; """
        f"margin: 2rem; background: #1e1e1e; color: #d4d4d4; }}"
        f"""
h1 {{ color: #569cd6; }}
.stats {{ margin: 1rem 0; color: #6a6a6a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 0.5rem 0.8rem; text-align: left; }}
th {{ background: #2d2d2d; color: #569cd6; }}
tr:nth-child(even) {{ background: #252525; }}
</style>
</head>
<body>
<h1>AgenticCLI Conversation History</h1>
<p class="stats">{total_count} records, {total_turns} turns</p>
<table>
<tr><th>Summary</th><th>Time</th><th>Turns</th></tr>
{''.join(rows)}
</table>
</body>
</html>"""
    )


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
