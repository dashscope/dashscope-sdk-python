# -*- coding: utf-8 -*-
"""Miscellaneous command handlers (trust, history, report)."""
# pylint: disable=protected-access,too-many-branches,too-many-statements
# pylint: disable=too-many-return-statements

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markup import escape

from dashscope.acli.agent import Agent

console = Console()


def _handle_trust_command(cmd: str, agent: Agent) -> None:
    """Inspect / mutate the Executor's session-scoped trust cache."""
    parts = cmd.strip().split()
    ex = agent.executor
    sub = parts[1] if len(parts) > 1 else ""

    if not sub or sub == "list":
        if ex._always_allow:
            console.print(
                "[bold green]Trusted tools this session:[/bold green]",
            )
            for name in sorted(ex._always_allow):
                console.print(f"  ✓ {name}")
        if ex._always_deny:
            console.print("[bold red]Denied tools this session:[/bold red]")
            for name in sorted(ex._always_deny):
                console.print(f"  ✗ {name}")
        if not ex._always_allow and not ex._always_deny:
            console.print(
                "[dim]No trust/deny cache this session "
                "(auto-cleared when a new session starts)[/dim]",
            )
        console.print(
            "\n[dim]Usage:\n"
            "  /trust              — list cache\n"
            "  /trust clear        — clear now (no need to wait for "
            "the session to end)\n"
            "  /trust allow <tool> — pre-trust; applies to the next "
            "session only\n"
            "  /trust deny  <tool> — pre-deny; applies to the next "
            "session only\n"
            "                        (pressing [s]top in the prompt aborts "
            "this session; nothing is added to the deny cache)[/dim]",
        )
        return

    if sub == "clear":
        n = len(ex._always_allow) + len(ex._always_deny)
        ex._always_allow.clear()
        ex._always_deny.clear()
        console.print(f"[dim]Cleared {n} cache entries[/dim]")
        return

    if sub in ("allow", "deny") and len(parts) >= 3:
        tool = parts[2]
        if sub == "allow":
            ex._always_deny.discard(tool)
            ex._always_allow.add(tool)
            console.print(
                f"[green]✓ {tool} trusted "
                f"(applies to next session only)[/green]",
            )
        else:
            ex._always_allow.discard(tool)
            ex._always_deny.add(tool)
            console.print(
                f"[red]✗ {tool} denied "
                f"(applies to next session only)[/red]",
            )
        return

    console.print(
        "[dim]Usage: /trust [list|clear|allow <tool>|deny <tool>][/dim]",
    )


def _message_text(msg: dict[str, Any]) -> str:
    """Flatten a chat message's content into one-line plain text."""
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        return ""
    return " ".join(content.split())


def _search_snippet(text: str, idx: int, kw_len: int) -> str:
    """Build a short one-line snippet centered on a match position."""
    start = max(0, idx - 20)
    end = min(len(text), idx + kw_len + 40)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _highlight_keyword(text: str, keyword: str) -> str:
    """Wrap keyword occurrences in rich markup (case-insensitive)."""
    lower_kw = keyword.lower()
    if not lower_kw:
        return escape(text)
    lower_text = text.lower()
    out: list[str] = []
    pos = 0
    while True:
        idx = lower_text.find(lower_kw, pos)
        if idx < 0:
            out.append(escape(text[pos:]))
            break
        out.append(escape(text[pos:idx]))
        out.append("[bold yellow]")
        out.append(escape(text[idx : idx + len(keyword)]))
        out.append("[/bold yellow]")
        pos = idx + len(keyword)
    return "".join(out)


def _history_search_matches(
    keyword: str,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Case-insensitive substring search across all session history.

    Scans every stored session's messages and returns up to ``limit``
    matches, each carrying the session topic, a timestamp, the message
    role, and a one-line snippet with match context.
    """
    from dashscope.acli.session import get_session_manager

    needle = keyword.lower()
    if not needle or limit <= 0:
        return []
    mgr = get_session_manager()
    matches: list[dict[str, str]] = []
    for meta in mgr.list_topics():
        if len(matches) >= limit:
            break
        for msg in mgr.load_messages(meta.topic):
            text = _message_text(msg)
            idx = text.lower().find(needle) if text else -1
            if idx < 0:
                continue
            matches.append(
                {
                    "session": meta.topic,
                    "timestamp": meta.last_accessed or "",
                    "role": str(msg.get("role", "?")),
                    "snippet": _search_snippet(text, idx, len(needle)),
                },
            )
            if len(matches) >= limit:
                break
    return matches


def _handle_history_command(cmd: str) -> None:
    """Manage conversation history."""
    from dashscope.acli.platforms.local.history import (
        clear_history,
        export_history,
        list_history,
        stats,
    )

    parts = cmd.strip().split()
    if len(parts) < 2:
        s = stats()
        if not s or s["count"] == 0:
            console.print("[dim]No conversation history yet[/dim]")
        else:
            console.print("[bold]Conversation history stats:[/bold]")
            console.print(f"  Total conversations: {s['count']}")
            console.print(f"  Total turns: {s['total_turns']}")
            entries = list_history(limit=5)
            if entries:
                console.print("\n[bold]Last 5 conversations:[/bold]")
                for e in entries:
                    console.print(
                        f"  - {e.get('summary', '')} "
                        f"({e.get('created_at', '')})",
                    )
        console.print(
            "\n[dim]Usage:\n"
            "  /history stats                          — show stats\n"
            "  /history list [n]                       — list recent n\n"
            "  /history search <keyword> [limit]       — full-text "
            "search\n"
            "  /history export <file> [--format json|md|html]  — export\n"
            "  /history clear                           — clear "
            "history[/dim]",
        )
        return

    sub = parts[1].lower()

    if sub == "stats":
        s = stats()
        if not s or s["count"] == 0:
            console.print("[dim]No conversation history yet[/dim]")
        else:
            console.print("[bold]Conversation history stats:[/bold]")
            console.print(f"  Total conversations: {s['count']}")
            console.print(f"  Total turns: {s['total_turns']}")
        return

    if sub == "list":
        try:
            limit = int(parts[2]) if len(parts) >= 3 else 10
        except ValueError:
            console.print(
                "[red]Usage: /history list [n] "
                "(n must be an integer)[/red]",
            )
            return
        entries = list_history(limit=limit)
        if not entries:
            console.print("[dim]No conversation history yet[/dim]")
        else:
            console.print(f"[bold]Last {len(entries)} conversations:[/bold]")
            for i, e in enumerate(entries, 1):
                console.print(f"  {i}. {e.get('summary', '')}")
                console.print(
                    f"     {e.get('created_at', '')} | "
                    f"{e.get('turns', 0)} turns",
                )
        return

    if sub == "search":
        if len(parts) < 3:
            console.print(
                "[dim]Usage: /history search <keyword> [limit][/dim]",
            )
            return
        keyword = parts[2]
        limit = 20
        if len(parts) >= 4:
            try:
                limit = int(parts[3])
            except ValueError:
                limit = 0
        if limit <= 0:
            console.print(
                "[red]limit must be a positive integer[/red]",
            )
            return
        matches = _history_search_matches(keyword, limit=limit)
        if not matches:
            console.print(f"[dim]No matches for '{keyword}'[/dim]")
            return
        header = f"[bold]{len(matches)} match(es) for '{keyword}':[/bold]"
        console.print(header)
        for i, m in enumerate(matches, 1):
            ts = m["timestamp"][:16]
            head = (
                f"  {i}. [cyan]{m['session']}[/cyan] "
                f"[dim]{ts}[/dim] [bold]{m['role']}[/bold]"
            )
            console.print(head)
            snippet = _highlight_keyword(m["snippet"], keyword)
            console.print(f"     {snippet}")
        return

    if sub == "export" and len(parts) >= 3:
        output_path = parts[2]
        fmt = "html"
        if "--format" in parts:
            idx = parts.index("--format")
            if idx + 1 < len(parts):
                f = parts[idx + 1].lower()
                fmt = "markdown" if f in ("md", "markdown") else f
        elif output_path.endswith(".json"):
            fmt = "json"
        elif output_path.endswith((".md", ".markdown")):
            fmt = "markdown"
        try:
            resolved = export_history(output_path, fmt=fmt)
            console.print(
                f"[green]✓ History exported to: {resolved} "
                f"(format: {fmt})[/green]",
            )
        except Exception as e:
            console.print(f"[red]Export failed: {e}[/red]")
        return

    if sub == "clear":
        count = clear_history()
        console.print(f"[green]✓ Cleared {count} history records[/green]")
        return

    usage = "[dim]Usage: /history [stats|list|search|export|clear][/dim]"
    console.print(usage)


def _handle_report_command(agent: Agent) -> None:
    """Generate and display a performance report."""
    from dashscope.acli.memory.trace import generate_report

    report = generate_report(agent.trace_logger)
    if not report:
        console.print("[dim]Not enough trace data to generate a report[/dim]")
        return

    console.print("[bold]Performance report:[/bold]")
    console.print(f"  Total LLM calls: {report['total_llm_calls']}")
    console.print(f"  Total tool calls: {report['total_tool_calls']}")
    console.print(f"  Tool success rate: {report['tool_success_rate']:.1%}")
    console.print(f"  Avg response time: {report['avg_response_time']:.2f}s")

    if report.get("top_tools"):
        console.print("\n[bold]Top 5 tools:[/bold]")
        for name, count in report["top_tools"][:5]:
            console.print(f"  {name}: {count}")
