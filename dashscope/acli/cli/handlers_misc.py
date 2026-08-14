# -*- coding: utf-8 -*-
"""Miscellaneous command handlers (trust, history, report)."""
# pylint: disable=protected-access,too-many-branches,too-many-statements

from __future__ import annotations

from rich.console import Console

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

    console.print("[dim]Usage: /history [stats|list|export|clear][/dim]")


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
