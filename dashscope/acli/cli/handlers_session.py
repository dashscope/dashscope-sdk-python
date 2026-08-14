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
        console.print(f"[bold]Current session[/bold]: {current}")
        console.print(
            "\n[dim]Usage:\n"
            "  /session              — show current topic\n"
            "  /session new [name]   — new session (default: default)\n"
            "  /session list         — list all sessions\n"
            "  /session switch <n>   — switch to a topic\n"
            "  /session rename <old> <new> — rename\n"
            "  /session remove <n>   — remove session "
            "(default cannot be removed)[/dim]",
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
            console.print(
                f"[green]Created and switched to new session: "
                f"{topic}[/green]",
            )
        else:
            console.print(
                f"[yellow]Session '{topic}' already exists; use "
                f"/session switch {topic} to switch[/yellow]",
            )

    elif subcmd == "list":
        topics = session_mgr.list_topics()
        if not topics:
            console.print("[dim]No sessions yet[/dim]")
        else:
            current = session_mgr.get_current_topic()
            console.print(f"[bold]Session list[/bold] ({len(topics)}):")
            for meta in topics:
                marker = " ← current" if meta.topic == current else ""
                accessed = (
                    meta.last_accessed[:16] if meta.last_accessed else ""
                )
                console.print(f"  • [cyan]{meta.topic}[/cyan]{marker}")
                console.print(
                    f"    [dim]{meta.message_count} messages, "
                    f"last accessed: {accessed}[/dim]",
                )

    elif subcmd == "switch":
        topic = parts[2] if len(parts) > 2 else ""
        if not topic:
            console.print("[dim]Usage: /session switch <topic>[/dim]")
            return

        if session_mgr.set_current_topic(topic):
            # Save current session
            if agent.session_path and agent.messages:
                agent.save_session()

            # Reset and load new topic
            agent.reset()
            agent.session_path = session_mgr.get_history_path(topic)
            restored = agent.load_session()
            console.print(f"[green]Switched to session: {topic}[/green]")
            if restored:
                console.print(
                    f"  [dim]Restored {restored} " f"history messages[/dim]",
                )
        else:
            console.print(
                f"[red]Session '{topic}' does not exist; "
                f"see /session list for available sessions[/red]",
            )

    elif subcmd == "rename":
        if len(parts) < 3:
            console.print("[dim]Usage: /session rename <old> <new>[/dim]")
            return
        old_new = parts[2].split(maxsplit=1)
        if len(old_new) < 2:
            console.print("[dim]Usage: /session rename <old> <new>[/dim]")
            return
        old_name, new_name = old_new

        was_current = session_mgr.get_current_topic() == old_name
        if session_mgr.rename_topic(old_name, new_name):
            # rename_topic already moved the current-topic pointer; also
            # repoint the live agent's session file so history keeps
            # appending to the renamed topic instead of the old path.
            if was_current and agent.session_path:
                agent.session_path = session_mgr.get_history_path(new_name)
            console.print(f"[green]Renamed: {old_name} → {new_name}[/green]")
        else:
            console.print(
                f"[red]Rename failed: '{old_name}' does not exist "
                f"or '{new_name}' already exists[/red]",
            )

    elif subcmd == "remove":
        topic = parts[2] if len(parts) > 2 else ""
        if not topic:
            console.print("[dim]Usage: /session remove <topic>[/dim]")
            return

        if topic == "default":
            console.print("[yellow]Cannot delete the default session[/yellow]")
            return

        if session_mgr.delete_topic(topic):
            console.print(f"[green]Removed session: {topic}[/green]")
        else:
            console.print(f"[red]Session '{topic}' does not exist[/red]")

    else:
        console.print(
            "[dim]Usage:\n"
            "  /session              — show current topic\n"
            "  /session new [name]   — new session (default: default)\n"
            "  /session list         — list all sessions\n"
            "  /session switch <n>   — switch to a topic\n"
            "  /session rename <old> <new> — rename\n"
            "  /session remove <n>   — remove session "
            "(default cannot be removed)[/dim]",
        )
