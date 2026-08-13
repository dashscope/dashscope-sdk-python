"""Profile and memory command handlers."""

from __future__ import annotations

from rich.console import Console

from dashscope.acli.config import Config

console = Console()

# Shared memory client (set by _run_loop / _run_tui_mode)
_memory_client = None


def set_memory_client(client) -> None:
    """Set the shared memory client instance."""
    global _memory_client
    _memory_client = client


async def _handle_profile_command(cmd: str, config: Config):
    """Handle /profile commands (async)."""
    global _memory_client
    parts = cmd.strip().split(maxsplit=2)

    if not _memory_client:
        if not config.memory_enabled:
            console.print(
                "[dim]用户档案已禁用。在 config.toml 中设置 memory_enabled = true 启用[/dim]"
            )
            return
        from dashscope.acli.platforms import get_memory_provider

        _memory_client = get_memory_provider(config)
        if not _memory_client:
            console.print("[red]未配置 API Key 或用户名，无法使用用户档案[/red]")
            return

    if len(parts) == 1:
        console.print(
            f"[bold]用户档案[/bold]: {'启用' if config.memory_enabled else '禁用'}"
        )
        if config.user_name:
            console.print(f"[dim]用户: {config.user_name}[/dim]")
        if config.memory_user_id:
            console.print(f"[dim]记忆标识: {config.memory_user_id}[/dim]")
        if config.memory_library_id:
            console.print(f"[dim]档案库: {config.memory_library_id}[/dim]")
        console.print(
            "\n[dim]用法:\n"
            "  /profile            — 显示状态\n"
            "  /profile list       — 查看档案\n"
            "  /profile search <q> — 搜索档案\n"
            "  /profile add <text>    — 添加信息\n"
            "  /profile remove <num>  — 删除指定档案\n"
            "  /profile clear         — 清空档案[/dim]\n"
            "\n说明: 对话中提到的个人信息（技术栈、偏好等）会自动提取保存。"
        )
        return

    subcmd = parts[1]

    if subcmd == "list":
        try:
            nodes = await _memory_client.list(page_size=20)
            if not nodes:
                console.print("[dim]暂无档案信息[/dim]")
            else:
                console.print(f"[bold]用户档案[/bold] ({len(nodes)} 条):")
                for i, n in enumerate(nodes, 1):
                    time_str = n.updated_at or n.created_at
                    console.print(f"  {i}. {n.content}")
                    console.print(f"     [dim]{time_str} | id: {n.id[:8]}...[/dim]")
        except Exception as e:
            console.print(f"[red]获取档案失败: {e}[/red]")

    elif subcmd == "search":
        query = parts[2] if len(parts) > 2 else ""
        if not query:
            console.print("[dim]用法: /profile search <关键词>[/dim]")
            return
        try:
            nodes = await _memory_client.search(query, top_k=5, min_score=0.3)
            if not nodes:
                console.print("[dim]未找到相关信息[/dim]")
            else:
                console.print(f"[bold]搜索结果[/bold] ({len(nodes)} 条):")
                for i, n in enumerate(nodes, 1):
                    console.print(
                        f"  {i}. {n.content} [dim](score: {n.score:.2f})[/dim]"
                    )
        except Exception as e:
            console.print(f"[red]搜索失败: {e}[/red]")

    elif subcmd == "add":
        content = parts[2] if len(parts) > 2 else ""
        if not content:
            console.print("[dim]用法: /profile add <信息内容>[/dim]")
            return
        try:
            nodes = await _memory_client.add([], custom_content=content)
            if nodes:
                console.print(f"[green]已保存 {len(nodes)} 条档案信息[/green]")
            else:
                console.print("[green]已提交[/green]")
        except Exception as e:
            console.print(f"[red]保存失败: {e}[/red]")

    elif subcmd == "remove" or subcmd == "rm":
        idx_str = parts[2] if len(parts) > 2 else ""
        if not idx_str:
            console.print("[red]用法: /profile remove <编号>[/red]")
            return
        try:
            idx = int(idx_str) - 1
            nodes = await _memory_client.list(page_size=100)
            if idx < 0 or idx >= len(nodes):
                console.print(f"[red]无效编号: {idx_str}[/red]")
                return
            target = nodes[idx]
            await _memory_client.delete(target.id)
            console.print(f"[green]✓ 已删除档案 ({idx_str})[/green]")
        except ValueError:
            console.print("[red]请输入数字编号[/red]")
        except Exception as e:
            console.print(f"[red]删除失败: {e}[/red]")

    elif subcmd == "clear":
        try:
            nodes = await _memory_client.list(page_size=100)
            if not nodes:
                console.print("[dim]暂无档案可清除[/dim]")
                return
            count = 0
            for n in nodes:
                try:
                    await _memory_client.delete(n.id)
                    count += 1
                except Exception:
                    pass
            console.print(f"[green]已清除 {count} 条档案信息[/green]")
        except Exception as e:
            console.print(f"[red]清除失败: {e}[/red]")

    else:
        console.print(
            "[dim]用法:\n"
            "  /profile            — 显示状态\n"
            "  /profile list       — 查看档案\n"
            "  /profile search <q> — 搜索档案\n"
            "  /profile add <text>    — 添加信息\n"
            "  /profile remove <num>  — 删除指定档案\n"
            "  /profile clear         — 清空档案[/dim]"
        )


async def _handle_memory_command(cmd: str):
    """Handle /memory commands for conversation history."""
    from dashscope.acli.platforms.local import history

    parts = cmd.strip().split(maxsplit=2)

    if len(parts) == 1:
        # Show status
        stats = history.stats()
        console.print(
            f"[bold]对话历史[/bold]: {stats['count']} 条记录, {stats['total_turns']} 轮对话"
        )
        console.print(
            "\n[dim]用法:\n"
            "  /memory               — 显示统计\n"
            "  /memory list [n]      — 查看最近 n 条 (默认 20)\n"
            "  /memory search <q>    — 搜索历史\n"
            "  /memory remove <id|num>— 删除指定记录(ID 或列表编号)\n"
            "  /memory clear         — 清空所有历史[/dim]\n"
            "\n说明: 每次对话结束后自动存储摘要，用于跨会话上下文回忆。"
        )
        return

    subcmd = parts[1]

    if subcmd == "list":
        limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 20
        entries = history.list_history(limit)
        if not entries:
            console.print("[dim]暂无对话历史[/dim]")
        else:
            console.print(f"[bold]最近 {len(entries)} 条对话[/bold]:")
            for i, entry in enumerate(entries, 1):
                created = entry.get("created_at", "")[:16]  # truncate ISO
                turns = entry.get("turns", 0)
                summary = entry.get("summary", "")
                console.print(f"  {i}. [dim]{created}[/dim] ({turns}轮) {summary}")
                console.print(f"     [dim]id: {entry['id']}[/dim]")

    elif subcmd == "search":
        query = parts[2] if len(parts) > 2 else ""
        if not query:
            console.print("[dim]用法: /memory search <关键词>[/dim]")
            return
        results = history.search_history(query)
        if not results:
            console.print("[dim]未找到相关历史[/dim]")
        else:
            console.print(f"[bold]搜索结果[/bold] ({len(results)} 条):")
            for i, entry in enumerate(results, 1):
                created = entry.get("created_at", "")[:16]
                console.print(f"  {i}. [dim]{created}[/dim] {entry['summary']}")

    elif subcmd == "remove" or subcmd == "rm":
        entry_id = parts[2] if len(parts) > 2 else ""
        if not entry_id:
            console.print("[dim]用法: /memory remove <id|num>[/dim]")
            return
        if history.delete_history(entry_id):
            console.print(f"[green]已删除记录 {entry_id}[/green]")
            return
        # Fallback: treat as 1-based index from /memory list
        try:
            idx = int(entry_id) - 1
            entries = history.list_history(20)
            if 0 <= idx < len(entries):
                target = entries[idx]
                if history.delete_history(target["id"]):
                    console.print(
                        f"[green]已删除记录 #{entry_id}: {target.get('summary', '')[:40]}[/green]"
                    )
                    return
        except ValueError:
            pass
        console.print(f"[red]未找到记录 {entry_id}[/red]")

    elif subcmd == "clear":
        count = history.clear_history()
        console.print(f"[green]已清除 {count} 条历史记录[/green]")

    else:
        console.print(
            "[dim]用法:\n"
            "  /memory               — 显示统计\n"
            "  /memory list [n]      — 查看最近 n 条\n"
            "  /memory search <q>    — 搜索历史\n"
            "  /memory remove <id|num>— 删除指定记录(ID 或列表编号)\n"
            "  /memory clear         — 清空所有历史[/dim]"
        )
