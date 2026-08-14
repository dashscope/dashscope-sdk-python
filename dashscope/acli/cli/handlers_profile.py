# -*- coding: utf-8 -*-
"""Profile and memory command handlers."""
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements

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
                "[dim]User profile is disabled. Set "
                "memory_enabled = true in config.toml to enable[/dim]",
            )
            return
        from dashscope.acli.platforms import get_memory_provider

        _memory_client = get_memory_provider(config)
        if not _memory_client:
            console.print(
                "[red]No API Key or username configured; "
                "user profile unavailable[/red]",
            )
            return

    if len(parts) == 1:
        console.print(
            f"[bold]User Profile[/bold]: "
            f"{'enabled' if config.memory_enabled else 'disabled'}",
        )
        if config.user_name:
            console.print(f"[dim]User: {config.user_name}[/dim]")
        if config.memory_user_id:
            console.print(f"[dim]Memory ID: {config.memory_user_id}[/dim]")
        if config.memory_library_id:
            console.print(f"[dim]Library: {config.memory_library_id}[/dim]")
        console.print(
            "\n[dim]Usage:\n"
            "  /profile            — show status\n"
            "  /profile list       — list entries\n"
            "  /profile search <q> — search profile\n"
            "  /profile add <text>    — add info\n"
            "  /profile remove <num>  — remove an entry\n"
            "  /profile clear         — clear profile[/dim]\n"
            "\nNote: personal info mentioned in chat (tech stack, "
            "preferences, etc.) is extracted and saved automatically.",
        )
        return

    subcmd = parts[1]

    if subcmd == "list":
        try:
            nodes = await _memory_client.list(page_size=20)
            if not nodes:
                console.print("[dim]No profile entries yet[/dim]")
            else:
                console.print(
                    f"[bold]User Profile[/bold] ({len(nodes)} entries):",
                )
                for i, n in enumerate(nodes, 1):
                    time_str = n.updated_at or n.created_at
                    console.print(f"  {i}. {n.content}")
                    console.print(
                        f"     [dim]{time_str} | id: {n.id[:8]}...[/dim]",
                    )
        except Exception as e:
            console.print(f"[red]Failed to load profile: {e}[/red]")

    elif subcmd == "search":
        query = parts[2] if len(parts) > 2 else ""
        if not query:
            console.print("[dim]Usage: /profile search <keyword>[/dim]")
            return
        try:
            nodes = await _memory_client.search(query, top_k=5, min_score=0.3)
            if not nodes:
                console.print("[dim]No matching info found[/dim]")
            else:
                console.print(
                    f"[bold]Search results[/bold] ({len(nodes)} entries):",
                )
                for i, n in enumerate(nodes, 1):
                    console.print(
                        f"  {i}. {n.content} "
                        f"[dim](score: {n.score:.2f})[/dim]",
                    )
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")

    elif subcmd == "add":
        content = parts[2] if len(parts) > 2 else ""
        if not content:
            console.print("[dim]Usage: /profile add <content>[/dim]")
            return
        try:
            nodes = await _memory_client.add([], custom_content=content)
            if nodes:
                console.print(
                    f"[green]Saved {len(nodes)} profile entries[/green]",
                )
            else:
                console.print("[green]Submitted[/green]")
        except Exception as e:
            console.print(f"[red]Save failed: {e}[/red]")

    elif subcmd in ("remove", "rm"):
        idx_str = parts[2] if len(parts) > 2 else ""
        if not idx_str:
            console.print("[red]Usage: /profile remove <num>[/red]")
            return
        try:
            idx = int(idx_str) - 1
            nodes = await _memory_client.list(page_size=100)
            if idx < 0 or idx >= len(nodes):
                console.print(f"[red]Invalid number: {idx_str}[/red]")
                return
            target = nodes[idx]
            await _memory_client.delete(target.id)
            console.print(f"[green]✓ Entry removed ({idx_str})[/green]")
        except ValueError:
            console.print("[red]Please enter a numeric index[/red]")
        except Exception as e:
            console.print(f"[red]Delete failed: {e}[/red]")

    elif subcmd == "clear":
        try:
            nodes = await _memory_client.list(page_size=100)
            if not nodes:
                console.print("[dim]No profile entries to clear[/dim]")
                return
            count = 0
            for n in nodes:
                try:
                    await _memory_client.delete(n.id)
                    count += 1
                except Exception:
                    pass
            console.print(f"[green]Cleared {count} profile entries[/green]")
        except Exception as e:
            console.print(f"[red]Clear failed: {e}[/red]")

    else:
        console.print(
            "[dim]Usage:\n"
            "  /profile            — show status\n"
            "  /profile list       — list entries\n"
            "  /profile search <q> — search profile\n"
            "  /profile add <text>    — add info\n"
            "  /profile remove <num>  — remove an entry\n"
            "  /profile clear         — clear profile[/dim]",
        )


async def _handle_memory_command(cmd: str):
    """Handle /memory commands for conversation history."""
    from dashscope.acli.platforms.local import history

    parts = cmd.strip().split(maxsplit=2)

    if len(parts) == 1:
        # Show status
        stats = history.stats()
        console.print(
            f"[bold]Chat history[/bold]: {stats['count']} records, "
            f"{stats['total_turns']} turns",
        )
        console.print(
            "\n[dim]Usage:\n"
            "  /memory               — show stats\n"
            "  /memory list [n]      — show last n entries (default 20)\n"
            "  /memory search <q>    — search history\n"
            "  /memory remove <id|num>— remove a record (ID or list index)\n"
            "  /memory clear         — clear all history[/dim]\n"
            "\nNote: a summary is stored after each conversation for "
            "cross-session context recall.",
        )
        return

    subcmd = parts[1]

    if subcmd == "list":
        limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 20
        entries = history.list_history(limit)
        if not entries:
            console.print("[dim]No conversation history yet[/dim]")
        else:
            console.print(f"[bold]Last {len(entries)} conversations[/bold]:")
            for i, entry in enumerate(entries, 1):
                created = entry.get("created_at", "")[:16]  # truncate ISO
                turns = entry.get("turns", 0)
                summary = entry.get("summary", "")
                console.print(
                    f"  {i}. [dim]{created}[/dim] ({turns} turns) {summary}",
                )
                console.print(f"     [dim]id: {entry['id']}[/dim]")

    elif subcmd == "search":
        query = parts[2] if len(parts) > 2 else ""
        if not query:
            console.print("[dim]Usage: /memory search <keyword>[/dim]")
            return
        results = history.search_history(query)
        if not results:
            console.print("[dim]No matching history found[/dim]")
        else:
            console.print(
                f"[bold]Search results[/bold] ({len(results)} entries):",
            )
            for i, entry in enumerate(results, 1):
                created = entry.get("created_at", "")[:16]
                console.print(
                    f"  {i}. [dim]{created}[/dim] {entry['summary']}",
                )

    elif subcmd in ("remove", "rm"):
        entry_id = parts[2] if len(parts) > 2 else ""
        if not entry_id:
            console.print("[dim]Usage: /memory remove <id|num>[/dim]")
            return
        if history.delete_history(entry_id):
            console.print(f"[green]Deleted record {entry_id}[/green]")
            return
        # Fallback: treat as 1-based index from /memory list
        try:
            idx = int(entry_id) - 1
            entries = history.list_history(20)
            if 0 <= idx < len(entries):
                target = entries[idx]
                if history.delete_history(target["id"]):
                    console.print(
                        f"[green]Deleted record #{entry_id}: "
                        f"{target.get('summary', '')[:40]}[/green]",
                    )
                    return
        except ValueError:
            pass
        console.print(f"[red]Record {entry_id} not found[/red]")

    elif subcmd == "clear":
        count = history.clear_history()
        console.print(f"[green]Cleared {count} history records[/green]")

    else:
        console.print(
            "[dim]Usage:\n"
            "  /memory               — show stats\n"
            "  /memory list [n]      — show last n entries\n"
            "  /memory search <q>    — search history\n"
            "  /memory remove <id|num>— remove a record (ID or list index)\n"
            "  /memory clear         — clear all history[/dim]",
        )
