"""Miscellaneous command handlers (trust, history, report)."""

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
            console.print("[bold green]本次对话已授信工具:[/bold green]")
            for name in sorted(ex._always_allow):
                console.print(f"  ✓ {name}")
        if ex._always_deny:
            console.print("[bold red]本次对话已拒绝工具:[/bold red]")
            for name in sorted(ex._always_deny):
                console.print(f"  ✗ {name}")
        if not ex._always_allow and not ex._always_deny:
            console.print(
                "[dim]本次对话无授信/拒绝缓存（每次新对话开始时自动清空）[/dim]"
            )
        console.print(
            "\n[dim]用法:\n"
            "  /trust              — 列出缓存\n"
            "  /trust clear        — 立即清空（无需等本轮结束）\n"
            "  /trust allow <tool> — 预先加入授信，仅对下一轮对话生效\n"
            "  /trust deny  <tool> — 预先加入拒绝，仅对下一轮对话生效\n"
            "                        (弹窗里按 [s]top 是中止本轮，不会进 deny 缓存)[/dim]"
        )
        return

    if sub == "clear":
        n = len(ex._always_allow) + len(ex._always_deny)
        ex._always_allow.clear()
        ex._always_deny.clear()
        console.print(f"[dim]已清空 {n} 条缓存[/dim]")
        return

    if sub in ("allow", "deny") and len(parts) >= 3:
        tool = parts[2]
        if sub == "allow":
            ex._always_deny.discard(tool)
            ex._always_allow.add(tool)
            console.print(f"[green]✓ {tool} 加入授信（仅对下一轮对话生效）[/green]")
        else:
            ex._always_allow.discard(tool)
            ex._always_deny.add(tool)
            console.print(f"[red]✗ {tool} 加入拒绝（仅对下一轮对话生效）[/red]")
        return

    console.print("[dim]用法: /trust [list|clear|allow <tool>|deny <tool>][/dim]")


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
            console.print("[dim]暂无对话历史[/dim]")
        else:
            console.print("[bold]对话历史统计:[/bold]")
            console.print(f"  总对话数: {s['count']}")
            console.print(f"  总轮数: {s['total_turns']}")
            entries = list_history(limit=5)
            if entries:
                console.print("\n[bold]最近 5 次对话:[/bold]")
                for e in entries:
                    console.print(
                        f"  - {e.get('summary', '')} ({e.get('created_at', '')})"
                    )
        console.print(
            "\n[dim]用法:\n"
            "  /history stats                          — 显示统计\n"
            "  /history list [n]                       — 列出最近 n 条\n"
            "  /history export <file> [--format json|md|html]  — 导出历史\n"
            "  /history clear                           — 清空历史[/dim]"
        )
        return

    sub = parts[1].lower()

    if sub == "stats":
        s = stats()
        if not s or s["count"] == 0:
            console.print("[dim]暂无对话历史[/dim]")
        else:
            console.print("[bold]对话历史统计:[/bold]")
            console.print(f"  总对话数: {s['count']}")
            console.print(f"  总轮数: {s['total_turns']}")
        return

    if sub == "list":
        try:
            limit = int(parts[2]) if len(parts) >= 3 else 10
        except ValueError:
            console.print("[red]用法: /history list [n]（n 为整数）[/red]")
            return
        entries = list_history(limit=limit)
        if not entries:
            console.print("[dim]暂无对话历史[/dim]")
        else:
            console.print(f"[bold]最近 {len(entries)} 次对话:[/bold]")
            for i, e in enumerate(entries, 1):
                console.print(f"  {i}. {e.get('summary', '')}")
                console.print(
                    f"     {e.get('created_at', '')} | {e.get('turns', 0)} 轮"
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
            console.print(f"[green]✓ 历史已导出到: {resolved} (format: {fmt})[/green]")
        except Exception as e:
            console.print(f"[red]导出失败: {e}[/red]")
        return

    if sub == "clear":
        count = clear_history()
        console.print(f"[green]✓ 已清空 {count} 条对话历史[/green]")
        return

    console.print("[dim]用法: /history [stats|list|export|clear][/dim]")


def _handle_report_command(agent: Agent) -> None:
    """Generate and display a performance report."""
    from dashscope.acli.memory.trace import generate_report

    report = generate_report(agent.trace_logger)
    if not report:
        console.print("[dim]暂无足够的 trace 数据生成报告[/dim]")
        return

    console.print("[bold]性能报告:[/bold]")
    console.print(f"  总 LLM 调用: {report['total_llm_calls']}")
    console.print(f"  总工具调用: {report['total_tool_calls']}")
    console.print(f"  工具成功率: {report['tool_success_rate']:.1%}")
    console.print(f"  平均响应时间: {report['avg_response_time']:.2f}s")

    if report.get("top_tools"):
        console.print("\n[bold]Top 5 工具:[/bold]")
        for name, count in report["top_tools"][:5]:
            console.print(f"  {name}: {count}")
