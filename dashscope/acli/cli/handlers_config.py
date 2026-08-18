# -*- coding: utf-8 -*-
"""Configuration-related command handlers (privacy, theme, directives,
rule)."""
# pylint: disable=too-many-branches,too-many-return-statements
# pylint: disable=too-many-statements,unused-argument

from __future__ import annotations

import copy
import json

from rich.console import Console

from dashscope.acli.cli.constants import THEME_PRESETS
from dashscope.acli.config import Config

console = Console()


def _handle_privacy_command(cmd: str, config: Config) -> None:
    """Manage privacy mode."""
    parts = cmd.strip().split()

    def _print_status():
        status = "🔒 Enabled" if config.privacy_mode else "🔓 Disabled"
        console.print(f"[bold]Privacy mode:[/bold] {status}")
        if config.privacy_mode:
            console.print(
                "  [dim]✓ Cloud capabilities disabled "
                "(profile, kb, cloud memory, mcp)[/dim]",
            )
            console.print("  [dim]✓ Scheduled tasks (cron) paused[/dim]")
            console.print(
                "  [dim]✓ Audit log redacted (tool args and user input "
                "truncated)[/dim]",
            )
            console.print(
                "  [yellow]⚠ LLM API calls are uncontrolled; "
                "conversation content is still sent to the model[/yellow]",
            )

    if len(parts) < 2:
        _print_status()
        return

    sub = parts[1].lower()

    if sub == "on":
        config.privacy_mode = True
        config.save_workspace()
        from dashscope.acli.audit import get_audit_logger

        get_audit_logger().set_privacy_mode(True)
        # Enforce at the tool surface too (not just the slash-command gate):
        # drop every registered cloud capability tool + connected MCP tools.
        from dashscope.acli.tools.platform import (
            unregister_cloud_capability_tools,
        )
        from dashscope.acli.tools.registry import registry

        removed = unregister_cloud_capability_tools()
        mcp_names = [
            t.name for t in registry.list_tools() if t.name.startswith("mcp_")
        ]
        for name in mcp_names:
            registry.unregister(name)
        console.print("[green]✓ Privacy mode enabled[/green]")
        console.print(
            "  [dim]Cloud capabilities disabled · cron paused · "
            "audit redacted[/dim]",
        )
        if removed or mcp_names:
            console.print(
                f"  [dim]Revoked {removed + len(mcp_names)} "
                f"cloud tools[/dim]",
            )
        console.print("  [yellow]Note: LLM API calls uncontrolled[/yellow]")
        return

    if sub == "off":
        config.privacy_mode = False
        config.save_workspace()
        from dashscope.acli.audit import get_audit_logger

        get_audit_logger().set_privacy_mode(False)
        # Re-register platform tools so enabled capabilities (incl. cloud)
        # are usable again without a restart.
        from dashscope.acli.cli.mcp import _connect_mcp
        from dashscope.acli.tools.platform import register_platform_tools

        register_platform_tools(config, connect_mcp_fn=_connect_mcp)
        console.print("[green]✓ Privacy mode disabled[/green]")
        return

    if sub == "status":
        _print_status()
        return

    console.print("[dim]Usage: /privacy [on|off|status][/dim]")


def _handle_debug_command(cmd: str, config: Config) -> None:
    """Toggle debug mode: record full LLM request payloads in the session
    trace."""
    from dashscope.acli import debuglog

    parts = cmd.strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "status"

    if sub == "on":
        config.debug = True
        config.save_workspace()
        debuglog.set_debug_enabled(True)
        console.print("[green]✓ Debug mode enabled[/green]")
        console.print(
            "  [dim]Full prompts will be recorded to the session trace "
            "(/log, /trace to view)[/dim]",
        )
        return

    if sub == "off":
        config.debug = False
        config.save_workspace()
        debuglog.set_debug_enabled(False)
        console.print("[green]✓ Debug mode disabled[/green]")
        return

    if sub == "status":
        status = "Enabled" if config.debug else "Disabled"
        console.print(f"[bold]Debug mode:[/bold] {status}")
        if config.debug:
            console.print(
                "  [dim]Full prompts of every LLM call are recorded to "
                "the session trace (/log, /trace to view)[/dim]",
            )
        return

    console.print("[dim]Usage: /debug [on|off|status][/dim]")


def _messages_text(msgs: list, *, full: bool) -> str:
    lines: list[str] = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if not full and len(content) > 500:
            content = content[:500] + " …(truncated)"
        lines.append(f"\n[{role}]\n{content}")
        tool_calls = m.get("tool_calls")
        if tool_calls:
            calls = json.dumps(tool_calls, ensure_ascii=False, default=str)
            lines.append(f"  tool_calls: {calls}")
    return "\n".join(lines)


def _usage_text(usage) -> str:
    if not isinstance(usage, dict):
        return ""
    parts = []
    inp = usage.get("input_tokens") or usage.get("prompt_tokens")
    if inp:
        s = f"↑{inp}"
        if usage.get("cached_tokens"):
            s += f" ({usage['cached_tokens']} cached)"
        parts.append(s)
    out = usage.get("output_tokens") or usage.get("completion_tokens")
    if out:
        parts.append(f"↓{out}")
    return " ".join(parts)


def _log_record_text(rec: dict, idx: int, *, full: bool) -> str:
    """Render one llm_call trace event; truncate message bodies unless full."""
    ts = str(rec.get("timestamp", ""))[:19].replace("T", " ")
    model = rec.get("model", "?")
    msgs = rec.get("messages", [])
    header = f"━━━ #{idx} {ts}  model={model}  messages={len(msgs)}"
    dur = rec.get("duration_ms", 0)
    if dur:
        header += f"  {dur / 1000:.1f}s"
    usage = _usage_text(rec.get("usage"))
    if usage:
        header += f"  {usage}"
    header += " ━━━"
    return header + _messages_text(msgs, full=full)


def _can_page() -> bool:
    """True only when the module console writes to a real terminal (CLI mode).

    In the TUI the console is captured into a buffer, and a pager would fight
    Textual for the screen.
    """
    import sys

    return console.file is sys.stdout and sys.stdout.isatty()


def _page_text(text: str) -> None:
    """Page text through less (scroll ↑/↓, search with /); print inline
    if unavailable.

    rich's console.pager() routes through pydoc, whose no-arg `(less)` probe
    fails on macOS ("Missing filename") and degrades to a plain dump — so
    invoke less directly on piped stdin instead.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["less", "-R"],
            input=text.encode("utf-8"),
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError):
        proc = None
    if proc is None or proc.returncode != 0:
        console.print(text, markup=False, highlight=False)


def _trace_events(agent, event_type: str | None = None) -> list[dict]:
    tracer = (
        getattr(agent, "trace_logger", None) if agent is not None else None
    )
    if tracer is None:
        return []
    events = list(tracer.iter_events())
    if event_type is not None:
        events = [e for e in events if e.get("event") == event_type]
    return events


def _handle_log_command(cmd: str, agent=None) -> None:
    """View recorded LLM request prompts: /log [tail [N] | search <kw> |
    clear].

    Data source: llm_call trace events that carry the full message payload,
    which are only recorded while debug mode is on (/debug on).
    """
    parts = cmd.strip().split(None, 2)
    sub = parts[1] if len(parts) > 1 else "tail"

    tracer = (
        getattr(agent, "trace_logger", None) if agent is not None else None
    )

    if sub == "clear":
        if tracer is not None:
            tracer.clear()
        console.print("[green]✓ Session trace cleared[/green]")
        return

    events = [e for e in _trace_events(agent, "llm_call") if e.get("messages")]

    if sub == "search":
        keyword = parts[2].strip() if len(parts) > 2 else ""
        if not keyword:
            console.print("[yellow]Usage: /log search <keyword>[/yellow]")
            return
        kw = keyword.lower()
        events = [
            e
            for e in events
            if kw in json.dumps(e, ensure_ascii=False, default=str).lower()
        ]
        header = f'LLM request search "{keyword}" ({len(events)} hits)'
    elif sub == "tail":
        limit = 3
        if len(parts) > 2:
            try:
                limit = int(parts[2].strip())
                if limit <= 0:
                    raise ValueError
            except ValueError:
                console.print(
                    "[yellow]Usage: /log tail [N] "
                    "(N = positive integer)[/yellow]",
                )
                return
        events = events[-limit:]
        header = f"Last {len(events)} LLM requests"
    else:
        console.print(
            "[dim]Usage: /log [tail [N] | search <keyword> | clear][/dim]",
        )
        return

    if not events:
        console.print(
            "[dim]No full prompts recorded yet "
            "(/debug on, then ask again)[/dim]",
        )
        return

    path = getattr(tracer, "trace_file", None)
    try:
        size_kb = path.stat().st_size // 1024 if path else 0
    except OSError:
        size_kb = 0

    if _can_page():
        text = f"{header}  ({path}, {size_kb} KB)\n\n" + "\n\n".join(
            _log_record_text(r, i + 1, full=True) for i, r in enumerate(events)
        )
        _page_text(text)
    else:
        console.print(
            f"[bold]{header}[/bold] [dim]({path}, {size_kb} KB)[/dim]",
        )
        for i, r in enumerate(events):
            console.print(_log_record_text(r, i + 1, full=False), markup=False)
        console.print(
            "[dim]Content truncated; page full logs in CLI mode[/dim]",
        )


def _trace_event_text(ev: dict, idx: int, *, full: bool) -> str:
    """Render one trace event compactly; full mode expands llm_call prompts."""
    elapsed = ev.get("elapsed_ms", 0)
    stamp = f"[+{elapsed / 1000:.1f}s]"
    event = ev.get("event", "?")

    if event == "llm_call":
        parts = [f"━━━ #{idx} {stamp} llm_call"]
        if ev.get("model"):
            parts.append(f"model={ev['model']}")
        parts.append(f"msgs={ev.get('message_count', 0)}")
        usage = _usage_text(ev.get("usage"))
        if usage:
            parts.append(usage)
        dur = ev.get("duration_ms", 0)
        if dur:
            parts.append(f"{dur / 1000:.1f}s")
        tcs = ev.get("tool_calls") or []
        if tcs:
            names = ", ".join(
                str(t.get("name", "?")) for t in tcs if isinstance(t, dict)
            )
            if names:
                parts.append(f"→ {names}")
        header = "  ".join(parts) + " ━━━"
        msgs = ev.get("messages")
        if msgs:
            return header + _messages_text(msgs, full=full)
        return header

    if event == "tool_execution":
        ok = "✓" if ev.get("success") else "✗"
        dur = ev.get("duration_ms", 0)
        args = json.dumps(
            ev.get("arguments", {}),
            ensure_ascii=False,
            default=str,
        )
        if not full and len(args) > 120:
            args = args[:120] + " …"
        preview = ev.get("result_preview", "")
        if not full and len(preview) > 120:
            preview = preview[:120] + " …"
        return (
            f"━━━ #{idx} {stamp} tool {ev.get('tool', '?')} · {dur}ms {ok}\n"
            f"  args: {args}\n  result: {preview}"
        )

    if event == "decision":
        details = {
            k: v
            for k, v in ev.items()
            if k not in ("timestamp", "elapsed_ms", "event")
        }
        body = json.dumps(details, ensure_ascii=False, default=str)
        return f"━━━ #{idx} {stamp} decision {ev.get('type', '?')} · {body}"

    body = json.dumps(
        {
            k: v
            for k, v in ev.items()
            if k not in ("timestamp", "elapsed_ms", "event")
        },
        ensure_ascii=False,
        default=str,
    )
    return f"━━━ #{idx} {stamp} {event} · {body}"


def _handle_trace_command(cmd: str, agent=None) -> None:
    """View the session execution trace: /trace [tail [N] | search <kw> |
    clear].

    Lightweight events (LLM call latency/usage, tool call latency,
    decision points) are always recorded; full prompts need /debug on.
    """
    parts = cmd.strip().split(None, 2)
    sub = parts[1] if len(parts) > 1 else "tail"

    tracer = (
        getattr(agent, "trace_logger", None) if agent is not None else None
    )

    if sub == "clear":
        if tracer is not None:
            tracer.clear()
        console.print("[green]✓ Session trace cleared[/green]")
        return

    if sub == "search":
        keyword = parts[2].strip() if len(parts) > 2 else ""
        if not keyword:
            console.print("[yellow]Usage: /trace search <keyword>[/yellow]")
            return
        events = tracer.search_events(keyword) if tracer is not None else []
        header = f'Trace search "{keyword}" ({len(events)} hits)'
    elif sub == "tail":
        limit = 5
        if len(parts) > 2:
            try:
                limit = int(parts[2].strip())
                if limit <= 0:
                    raise ValueError
            except ValueError:
                console.print(
                    "[yellow]Usage: /trace tail [N] "
                    "(N = positive integer)[/yellow]",
                )
                return
        events = tracer.tail_events(limit) if tracer is not None else []
        header = f"Last {len(events)} trace events"
    else:
        console.print(
            "[dim]Usage: /trace [tail [N] | search <keyword> | "
            "clear][/dim]",
        )
        return

    if not events:
        console.print(
            "[dim]No trace records yet (recorded automatically once "
            "this conversation makes calls)[/dim]",
        )
        return

    path = getattr(tracer, "trace_file", None)
    try:
        size_kb = path.stat().st_size // 1024 if path else 0
    except OSError:
        size_kb = 0

    if _can_page():
        text = f"{header}  ({path}, {size_kb} KB)\n\n" + "\n\n".join(
            _trace_event_text(e, i + 1, full=True)
            for i, e in enumerate(events)
        )
        _page_text(text)
    else:
        console.print(
            f"[bold]{header}[/bold] [dim]({path}, {size_kb} KB)[/dim]",
        )
        for i, e in enumerate(events):
            console.print(
                _trace_event_text(e, i + 1, full=False),
                markup=False,
            )
        console.print(
            "[dim]Content truncated; page full trace in CLI mode[/dim]",
        )


def _handle_theme_command(cmd: str, config: Config) -> None:
    """Manage UI theme settings."""
    parts = cmd.strip().split(maxsplit=2)
    if len(parts) < 2:
        current_name = next(
            (
                name
                for name, preset in THEME_PRESETS.items()
                if preset == config.theme
            ),
            "custom",
        )
        console.print(f"[bold]Current theme:[/bold] {current_name}")
        console.print(
            f"[dim]Available presets: "
            f"{', '.join(THEME_PRESETS.keys())}[/dim]",
        )
        console.print(
            "[dim]Usage: /theme list | /theme set <name> | /theme "
            "<name> | /theme <key> <color>[/dim]",
        )
        return

    sub = parts[1].lower()

    if sub == "list":
        current_name = next(
            (
                name
                for name, preset in THEME_PRESETS.items()
                if preset == config.theme
            ),
            "custom",
        )
        console.print(f"[bold]Current theme:[/bold] {current_name}")
        console.print(
            f"[dim]Available presets: "
            f"{', '.join(THEME_PRESETS.keys())}[/dim]",
        )
        return

    if sub == "set":
        theme_name = parts[2].lower() if len(parts) > 2 else ""
        if not theme_name or theme_name not in THEME_PRESETS:
            console.print(f"[red]Unknown theme: {theme_name}[/red]")
            console.print(
                f"[dim]Available: " f"{', '.join(THEME_PRESETS.keys())}[/dim]",
            )
            return
        # Deep-copy: later `/theme <key> <color>` overrides mutate
        # config.theme, which must not alias the global preset.
        config.theme = copy.deepcopy(THEME_PRESETS[theme_name])
        config.save_workspace()
        console.print(f"[green]✓ Theme switched to: {theme_name}[/green]")
        console.print("[dim]Takes effect after restart[/dim]")
        return

    # Backward compatibility: /theme <name>
    theme_name = sub
    if theme_name in THEME_PRESETS:
        config.theme = copy.deepcopy(THEME_PRESETS[theme_name])
        config.save_workspace()
        console.print(f"[green]✓ Theme switched to: {theme_name}[/green]")
        console.print("[dim]Takes effect after restart[/dim]")
        return

    # Custom color key: /theme background #1e1e1e
    if len(parts) >= 3 and sub in config.theme:
        config.theme[sub] = parts[2]
        config.save_workspace()
        console.print(f"[green]✓ Set {sub} = {parts[2]}[/green]")
        console.print("[dim]Takes effect after restart[/dim]")
        return

    console.print(f"[red]Unknown theme or attribute: {theme_name}[/red]")
    console.print(
        f"[dim]Available presets: " f"{', '.join(THEME_PRESETS.keys())}[/dim]",
    )


def _handle_directives_command(cmd: str, config: Config) -> None:
    """Manage custom directives (rules injected into system prompt)."""
    parts = cmd.strip().split(maxsplit=1)
    if len(parts) < 2:
        if not config.user_directives:
            console.print("[dim]No custom directives yet[/dim]")
        else:
            console.print("[bold]Custom directives:[/bold]")
            for i, directive in enumerate(config.user_directives, 1):
                console.print(f"  {i}. {directive}")
        console.print(
            "\n[dim]Usage:\n"
            "  /directives add <text>       — add a directive\n"
            "  /directives rm <num>         — remove a directive\n"
            "  /directives clear            — clear all directives\n"
            "  /directives proposals        — view auto-learned proposals\n"
            "  /directives accept <id>      — accept proposal as a rule\n"
            "  /directives reject <id>      — reject a proposal[/dim]",
        )
        return

    sub = parts[1].strip()

    if sub.startswith("add "):
        text = sub[4:].strip()
        if not text:
            console.print("[red]Please provide directive text[/red]")
            return
        config.user_directives.append(text)
        config.save_workspace()
        console.print(
            f"[green]✓ Directive added "
            f"({len(config.user_directives)} total)[/green]",
        )
        return

    if sub.startswith("rm "):
        try:
            idx = int(sub[3:].strip()) - 1
            if 0 <= idx < len(config.user_directives):
                removed = config.user_directives.pop(idx)
                config.save_workspace()
                console.print(f"[green]✓ Removed: {removed}[/green]")
            else:
                console.print(f"[red]Invalid number: {idx + 1}[/red]")
        except ValueError:
            console.print("[red]Please provide a valid number[/red]")
        return

    if sub == "clear":
        count = len(config.user_directives)
        config.user_directives.clear()
        config.save_workspace()
        console.print(f"[green]✓ Cleared {count} directives[/green]")
        return

    if sub == "proposals":
        from dashscope.acli.memory.directives_learning import (
            list_proposed_directives,
        )

        proposals = list_proposed_directives("pending")
        if not proposals:
            console.print("[dim]No auto-learned proposals yet[/dim]")
        else:
            console.print("[bold]Behavior pattern proposals:[/bold]")
            for p in proposals:
                console.print(f"  [{p['id']}] {p['directive']}")
                console.print(f"    [dim]{p['rationale']}[/dim]")
        return

    if sub.startswith("accept "):
        proposal_id = sub[7:].strip()
        from dashscope.acli.memory.directives_learning import accept_directive

        if accept_directive(proposal_id, config):
            console.print(
                f"[green]✓ Accepted proposal {proposal_id} "
                f"as a rule[/green]",
            )
        else:
            console.print(
                f"[red]Proposal {proposal_id} not found or "
                f"already handled[/red]",
            )
        return

    if sub.startswith("reject "):
        proposal_id = sub[7:].strip()
        from dashscope.acli.memory.directives_learning import reject_directive

        if reject_directive(proposal_id):
            console.print(
                f"[yellow]✗ Rejected proposal {proposal_id}[/yellow]",
            )
        else:
            console.print(
                f"[red]Proposal {proposal_id} not found or "
                f"already handled[/red]",
            )
        return

    console.print(
        "[dim]Usage: /directives "
        "[add|rm|clear|proposals|accept|reject][/dim]",
    )


def _handle_rule_command(cmd: str, config: Config) -> None:
    """Manage workspace rules (.acli/rules.jsonl)."""
    from dashscope.acli.config import WORKSPACE_DIR

    rules_file = WORKSPACE_DIR / "rules.jsonl"
    parts = cmd.strip().split(maxsplit=1)

    def _load_entries() -> list[dict]:
        if not rules_file.exists():
            return []
        entries: list[dict] = []
        for line in rules_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _save_entries(entries: list[dict]) -> None:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        rules_file.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
            encoding="utf-8",
        )

    def _print_rules():
        entries = _load_entries()
        if not entries:
            console.print("[dim]No workspace rules configured yet[/dim]")
            return
        console.print("[bold]Workspace rules:[/bold]")
        for i, entry in enumerate(entries, 1):
            text = entry.get("text", "")
            enabled = (
                "[green]✓[/green]"
                if entry.get("enabled", True)
                else "[dim]✗[/dim]"
            )
            console.print(f"  {enabled} {i}. {text}")

    if len(parts) < 2:
        _print_rules()
        console.print(
            "\n[dim]Usage:\n"
            "  /rule list                 — list rules\n"
            "  /rule add <text>           — add a rule\n"
            "  /rule remove <num>         — remove a rule (alias: rm)\n"
            "  /rule edit                 — open rules file in an editor\n"
            "  /rule edit <num> <text>    — edit the given rule\n"
            "  /rule enable <num>         — enable a rule\n"
            "  /rule disable <num>        — disable a rule\n"
            "  /rule clear                — clear all rules[/dim]",
        )
        return

    sub = parts[1].strip()

    if sub == "list":
        _print_rules()
        return

    if sub.startswith("add "):
        text = sub[4:].strip()
        if not text:
            console.print("[red]Please provide rule text[/red]")
            return
        entries = _load_entries()
        entries.append({"text": text, "enabled": True})
        _save_entries(entries)
        console.print(f"[green]✓ Rule added: {text}[/green]")
        return

    if sub == "edit":
        if not rules_file.exists():
            WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            rules_file.write_text("", encoding="utf-8")
        import os

        editor = os.environ.get("EDITOR", "nano")
        os.system(f"{editor} {rules_file}")
        console.print("[green]✓ Rules file updated[/green]")
        return

    if sub.startswith("edit "):
        rest = sub[5:].strip()
        parts2 = rest.split(maxsplit=1)
        if len(parts2) < 2:
            console.print("[red]Usage: /rule edit <num> <new text>[/red]")
            return
        try:
            idx = int(parts2[0]) - 1
        except ValueError:
            console.print("[red]Please provide a valid number[/red]")
            return
        new_text = parts2[1]
        entries = _load_entries()
        if idx < 0 or idx >= len(entries):
            console.print(f"[red]Invalid number: {idx + 1}[/red]")
            return
        entries[idx]["text"] = new_text
        _save_entries(entries)
        console.print(f"[green]✓ Updated rule #{idx + 1}: {new_text}[/green]")
        return

    if sub.startswith("enable "):
        num_part = sub[7:].strip()
        try:
            idx = int(num_part) - 1
        except ValueError:
            console.print("[red]Please provide a valid number[/red]")
            return
        entries = _load_entries()
        if idx < 0 or idx >= len(entries):
            console.print(f"[red]Invalid number: {idx + 1}[/red]")
            return
        entries[idx]["enabled"] = True
        _save_entries(entries)
        console.print(f"[green]✓ Enabled rule #{idx + 1}[/green]")
        return

    if sub.startswith("disable "):
        num_part = sub[8:].strip()
        try:
            idx = int(num_part) - 1
        except ValueError:
            console.print("[red]Please provide a valid number[/red]")
            return
        entries = _load_entries()
        if idx < 0 or idx >= len(entries):
            console.print(f"[red]Invalid number: {idx + 1}[/red]")
            return
        entries[idx]["enabled"] = False
        _save_entries(entries)
        console.print(f"[green]✓ Disabled rule #{idx + 1}[/green]")
        return

    if (
        sub == "remove"
        or sub.startswith("remove ")
        or sub == "rm"
        or sub.startswith("rm ")
    ):
        entries = _load_entries()
        if not entries:
            console.print("[dim]No rules to remove[/dim]")
            return
        if sub.startswith("remove "):
            num_part = sub[7:].strip()
        elif sub.startswith("rm "):
            num_part = sub[3:].strip()
        else:
            num_part = ""
        if not num_part:
            console.print(
                "[red]Please provide a number, e.g. /rule remove 1 "
                "or /rule rm 1[/red]",
            )
            return
        try:
            idx = int(num_part) - 1
        except ValueError:
            console.print("[red]Please provide a valid number[/red]")
            return
        if idx < 0 or idx >= len(entries):
            console.print(f"[red]Invalid number: {idx + 1}[/red]")
            return
        removed = entries.pop(idx)
        _save_entries(entries)
        console.print(
            f"[green]✓ Removed rule: {removed.get('text', '')}[/green]",
        )
        return

    if sub == "clear":
        if rules_file.exists():
            rules_file.unlink()
            console.print("[green]✓ Rules cleared[/green]")
        else:
            console.print("[dim]No rules to clear[/dim]")
        return

    console.print(
        "[dim]Usage: /rule [list|add|remove(rm)|edit|edit <num> "
        "<text>|enable|disable|clear][/dim]",
    )
