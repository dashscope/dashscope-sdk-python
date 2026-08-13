# -*- coding: utf-8 -*-
"""Session management command handlers."""
# pylint: disable=too-many-branches,too-many-statements,unused-argument

from __future__ import annotations

from rich.console import Console

console = Console()


def _handle_session_command(cmd: str, config, agent) -> None:
    """Handle /session commands for multi-topic session management."""
    from dashscope.acli.session import get_session_manager

    session_mgr = get_session_manager()
    parts = cmd.strip().split(maxsplit=2)

    if len(parts) == 1:
        # Show current topic
        current = session_mgr.get_current_topic()
        console.print(f"[bold]当前会话[/bold]: {current}")
        console.print(
            "\n[dim]用法:\n"
            "  /session              — 显示当前主题\n"
            "  /session new [name]   — 新建会话（默认 default）\n"
            "  /session list         — 列出所有会话\n"
            "  /session switch <n>   — 切换到指定主题\n"
            "  /session rename <old> <new> — 重命名\n"
            "  /session remove <n>   — 删除会话（default 不可删）[/dim]",
        )
        return

    subcmd = parts[1]

    if subcmd == "new":
        topic = parts[2] if len(parts) > 2 else "default"
        # Archive current session first
        if agent.session_path and agent.messages:
            agent.save_session()

        if session_mgr.create_topic(topic):
            # Reset agent and switch to new topic
            agent.reset()
            agent.session_path = session_mgr.get_history_path(topic)
            console.print(f"[green]已创建并切换到新会话: {topic}[/green]")
        else:
            console.print(
                f"[yellow]会话 '{topic}' 已存在，使用 "
                f"/session switch {topic} 切换[/yellow]",
            )

    elif subcmd == "list":
        topics = session_mgr.list_topics()
        if not topics:
            console.print("[dim]暂无会话[/dim]")
        else:
            current = session_mgr.get_current_topic()
            console.print(f"[bold]会话列表[/bold] ({len(topics)} 个):")
            for meta in topics:
                marker = " ← 当前" if meta.topic == current else ""
                accessed = (
                    meta.last_accessed[:16] if meta.last_accessed else ""
                )
                console.print(f"  • [cyan]{meta.topic}[/cyan]{marker}")
                console.print(
                    f"    [dim]{meta.message_count} 条消息, "
                    f"最后访问: {accessed}[/dim]",
                )

    elif subcmd == "switch":
        topic = parts[2] if len(parts) > 2 else ""
        if not topic:
            console.print("[dim]用法: /session switch <主题名>[/dim]")
            return

        if session_mgr.set_current_topic(topic):
            # Save current session
            if agent.session_path and agent.messages:
                agent.save_session()

            # Reset and load new topic
            agent.reset()
            agent.session_path = session_mgr.get_history_path(topic)
            restored = agent.load_session()
            console.print(f"[green]已切换到会话: {topic}[/green]")
            if restored:
                console.print(f"  [dim]已恢复 {restored} 条历史消息[/dim]")
        else:
            console.print(
                f"[red]会话 '{topic}' 不存在，使用 /session list 查看可用会话[/red]",
            )

    elif subcmd == "rename":
        if len(parts) < 3:
            console.print("[dim]用法: /session rename <旧名> <新名>[/dim]")
            return
        old_new = parts[2].split(maxsplit=1)
        if len(old_new) < 2:
            console.print("[dim]用法: /session rename <旧名> <新名>[/dim]")
            return
        old_name, new_name = old_new

        was_current = session_mgr.get_current_topic() == old_name
        if session_mgr.rename_topic(old_name, new_name):
            # rename_topic already moved the current-topic pointer; also
            # repoint the live agent's session file so history keeps
            # appending to the renamed topic instead of the old path.
            if was_current and agent.session_path:
                agent.session_path = session_mgr.get_history_path(new_name)
            console.print(f"[green]已重命名: {old_name} → {new_name}[/green]")
        else:
            console.print(
                f"[red]重命名失败：'{old_name}' 不存在或 '{new_name}' 已存在[/red]",
            )

    elif subcmd == "remove":
        topic = parts[2] if len(parts) > 2 else ""
        if not topic:
            console.print("[dim]用法: /session remove <主题名>[/dim]")
            return

        if topic == "default":
            console.print("[yellow]不能删除 default 会话[/yellow]")
            return

        if session_mgr.delete_topic(topic):
            console.print(f"[green]已删除会话: {topic}[/green]")
        else:
            console.print(f"[red]会话 '{topic}' 不存在[/red]")

    else:
        console.print(
            "[dim]用法:\n"
            "  /session              — 显示当前主题\n"
            "  /session new [name]   — 新建会话（默认 default）\n"
            "  /session list         — 列出所有会话\n"
            "  /session switch <n>   — 切换到指定主题\n"
            "  /session rename <old> <new> — 重命名\n"
            "  /session remove <n>   — 删除会话（default 不可删）[/dim]",
        )
