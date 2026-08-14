# -*- coding: utf-8 -*-
"""Command dispatch functions."""
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements,too-many-nested-blocks

from __future__ import annotations

import sys

from rich.console import Console

from dashscope.acli.agent import Agent
from dashscope.acli.cli.handlers_config import (
    _handle_debug_command,
    _handle_directives_command,
    _handle_log_command,
    _handle_privacy_command,
    _handle_rule_command,
    _handle_theme_command,
    _handle_trace_command,
)
from dashscope.acli.cli.handlers_device import (
    _handle_camera_command,
    _handle_tts_command,
    _handle_voice_command,
)
from dashscope.acli.cli.handlers_misc import (
    _handle_history_command,
    _handle_report_command,
    _handle_trust_command,
)
from dashscope.acli.cli.handlers_session import _handle_session_command
from dashscope.acli.commands import render_help_text
from dashscope.acli.config import Config
from dashscope.acli.dev import handle_dev_command

console = Console()


def _handle_slash_command(
    cmd: str,
    agent: Agent,
    config: Config,
) -> bool | str:
    """Handle slash commands.

    Returns True if handled, 'async' if needs async handling, 'voice' for
    voice input, False otherwise.
    """
    cmd = cmd.strip()
    if cmd == "/v":
        return "voice"
    if cmd == "/voice" or cmd.startswith("/voice "):
        voice_result = _handle_voice_command(cmd, config)
        if voice_result == "voice":
            return "voice"
        return True
    if cmd == "/tts" or cmd.startswith("/tts "):
        _handle_tts_command(cmd, config, agent)
        return True
    if cmd == "/camera" or cmd.startswith("/camera "):
        _handle_camera_command(cmd)
        return True
    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]Bye![/dim]")
        sys.exit(0)
    elif cmd == "/copy":
        import dashscope.acli.cli as _cli

        _cons = _cli.console
        if not agent.last_output:
            _cons.print("[yellow]Nothing to copy[/yellow]")
            return True
        import shutil
        import subprocess as _sp

        if shutil.which("pbcopy"):
            _sp.run(["pbcopy"], input=agent.last_output.encode(), check=False)
        elif shutil.which("xclip"):
            _sp.run(
                ["xclip", "-selection", "clipboard"],
                input=agent.last_output.encode(),
                check=False,
            )
        elif shutil.which("clip"):
            _sp.run(["clip"], input=agent.last_output.encode(), check=False)
        else:
            _cons.print(
                "[yellow]No clipboard tool found "
                "(pbcopy/xclip/clip)[/yellow]",
            )
            return True
        _cons.print(
            f"[green]✓ Copied {len(agent.last_output)} "
            f"chars to clipboard[/green]",
        )
        return True
    elif cmd == "/json" or cmd.startswith("/json "):
        parts = cmd.strip().split()
        if len(parts) >= 2:
            arg = parts[1].lower()
            if arg in ("on", "true", "1"):
                agent.json_mode = True
                console.print(
                    "[green]✓ JSON output mode on "
                    "(replies forced to JSON)[/green]",
                )
            elif arg in ("off", "false", "0"):
                agent.json_mode = False
                console.print("[green]✓ JSON output mode off[/green]")
            else:
                console.print("[dim]Usage: /json on|off[/dim]")
        else:
            state = "on" if agent.json_mode else "off"
            console.print(f"[bold]JSON output mode: {state}[/bold]")
            console.print(
                "[dim]Usage: /json on|off — when on, "
                "replies are forced to JSON[/dim]",
            )
        return True
    elif cmd == "/save" or cmd.startswith("/save "):
        parts = cmd.strip().split(None, 1)
        if not agent.last_output:
            console.print("[yellow]Nothing to save[/yellow]")
            return True
        save_path = parts[1].strip() if len(parts) > 1 else None
        if not save_path:
            import datetime

            save_path = (
                f"acli_output_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
        try:
            from pathlib import Path

            p = Path(save_path)
            p.write_text(agent.last_output, encoding="utf-8")
            console.print(
                f"[green]✓ Saved {len(agent.last_output)} "
                f"chars to: {p}[/green]",
            )
        except Exception as e:
            console.print(f"[red]Save failed: {e}[/red]")
        return True
    elif cmd == "/clear":
        agent.reset()
        # Persist empty history so the cleared state survives restart.
        if agent.session_path:
            agent.save_session()
        console.print("[dim]Conversation cleared[/dim]")
        return True
    elif cmd == "/compress":
        if len(agent.messages) < 4:
            console.print(
                "[yellow]Conversation too short to compress[/yellow]",
            )
            return True
        return "compress"
    elif cmd == "/info":
        from dashscope.acli.config import CONFIG_FILE, WORKSPACE_CONFIG_FILE
        from dashscope.acli.extensions import (
            GLOBAL_EXTENSIONS_FILE,
            WORKSPACE_EXTENSIONS_FILE,
        )

        console.print()
        console.print("[bold]ℹ️ Current runtime info[/bold]")
        console.print(
            f"  Provider:     [cyan]"
            f"{agent.provider_name or config.provider}[/cyan]",
        )
        console.print(
            f"  Model:        [cyan]{agent.model_name or config.model}[/cyan]",
        )
        console.print(
            f"  Protocol:     [cyan]{config.protocol or 'auto'}[/cyan]",
        )
        # Resolve actual base URL from provider profile
        from dashscope.acli.providers.profile import build_profiles_from_config

        profiles = build_profiles_from_config(config)
        actual_base_url = profiles[0].base_url if profiles else config.base_url
        console.print(
            f"  Base URL:     [cyan]{actual_base_url or '(default)'}[/cyan]",
        )
        console.print(f"  User:         [cyan]{config.user_name}[/cyan]")
        console.print(
            f"  Memory:       "
            f"{'[green]on' if config.memory_enabled else '[dim]off'}[/]",
        )
        console.print(
            f"  Capabilities: [cyan]"
            f"{', '.join(config.enabled_capabilities or []) or '(none)'}"
            f"[/cyan]",
        )
        console.print(
            f"  JSON mode:    "
            f"{'[green]on' if agent.json_mode else '[dim]off'}[/]",
        )
        console.print(f"  Loop mode:    [cyan]{config.loop_mode}[/cyan]")
        console.print(f"  Max turns:    [cyan]{config.max_turns}[/cyan]")
        console.print(f"  Timeout:      [cyan]{config.timeout}s[/cyan]")
        console.print(
            f"  TUI:          {'[green]on' if config.tui else '[dim]off'}[/]",
        )
        console.print(
            f"  Privacy:      "
            f"{'[green]on' if config.privacy_mode else '[dim]off'}[/]",
        )

        console.print("\n[bold]Config files[/bold]")
        for label, path in [
            ("Global config", CONFIG_FILE),
            ("Workspace config", WORKSPACE_CONFIG_FILE),
            ("Global extensions", GLOBAL_EXTENSIONS_FILE),
            ("Workspace extensions", WORKSPACE_EXTENSIONS_FILE),
        ]:
            if path.exists():
                console.print(f"  [green]✓[/green] {label}: [dim]{path}[/dim]")
            else:
                console.print(f"  [dim]✗ {label}: {path}[/dim]")
        console.print()
        return True
    elif cmd == "/stats":
        stats = (
            agent.executor.get_stats() if hasattr(agent, "executor") else {}
        )
        console.print()
        console.print("[bold]📊 Session stats[/bold]")
        console.print(f"  Provider: [cyan]{agent.provider_name}[/cyan]")
        console.print(f"  Model:    [cyan]{agent.model_name}[/cyan]")

        # Token usage
        token_usage = stats.get("token_usage", {})
        if token_usage.get("total_tokens", 0) > 0:
            cached = token_usage.get("cached_tokens", 0)
            cached_part = f", cached: {cached}" if cached else ""
            console.print(
                f"  Tokens: [green]{token_usage['total_tokens']}[/green]"
                f" (in: {token_usage['input_tokens']},"
                f" out: {token_usage['output_tokens']}{cached_part})",
            )

        # Prompt composition (chars sent per LLM call, cumulative)
        prompt_comp = stats.get("prompt_composition", {})
        comp_total = sum(prompt_comp.values())
        if comp_total > 0:
            console.print(f"  Prompt mix: [green]{comp_total}[/green] chars")
            for key, label in [
                ("system", "system"),
                ("user", "user"),
                ("assistant", "assistant"),
                ("tools", "tools"),
            ]:
                v = prompt_comp.get(key, 0)
                if v:
                    console.print(
                        f"    [dim]{label}[/dim] → {v} "
                        f"({v * 100 // comp_total}%)",
                    )

        # API calls
        api_calls = stats.get("api_calls", 0)
        if api_calls > 0:
            console.print(f"  API calls: [blue]{api_calls}[/blue]")

        # Tool calls
        console.print(
            f"  Tool calls: [yellow]"
            f"{stats.get('total_tool_calls', 0)}[/yellow]",
        )
        tool_counts = stats.get("tool_counts", {})
        if tool_counts:
            console.print("  Tool breakdown:")
            for name, count in sorted(
                tool_counts.items(),
                key=lambda x: -x[1],
            ):
                console.print(f"    [dim]{name}[/dim] → {count}")

        # Skill activations
        skill_calls = stats.get("skill_calls", 0)
        if skill_calls > 0:
            console.print(f"  Skill calls: [magenta]{skill_calls}[/magenta]")
            skill_counts = stats.get("skill_counts", {})
            if skill_counts:
                console.print("  Skill breakdown:")
                for name, count in sorted(
                    skill_counts.items(),
                    key=lambda x: -x[1],
                ):
                    console.print(f"    [dim]{name}[/dim] → {count}")

        # Errors
        errors = stats.get("errors", 0)
        if errors > 0:
            console.print(f"  Errors: [red]{errors}[/red]")

        # Session duration
        duration = stats.get("session_duration", 0)
        if duration > 0:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            if minutes > 0:
                console.print(
                    f"  Duration: [magenta]{minutes}m{seconds}s[/magenta]",
                )
            else:
                console.print(f"  Duration: [magenta]{seconds}s[/magenta]")

        console.print()
        return True
    elif cmd in ("/feedback good", "/feedback bad"):
        outcome = "success" if "good" in cmd else "failure"
        tracker = getattr(agent, "experience_tracker", None)
        if tracker and agent.last_output:
            # Get recent tools used
            tools_used = []
            for msg in reversed(agent.messages):
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if "function" in tc:
                            tools_used.append(tc["function"]["name"])
                    break

            task_summary = (
                agent.messages[-2]["content"][:100]
                if len(agent.messages) > 1
                else "user task"
            )
            lesson = (
                "User feedback: satisfied"
                if outcome == "success"
                else "User feedback: unsatisfied"
            )
            tracker.record_experience(
                task_summary=task_summary,
                tools_used=tools_used,
                outcome=outcome,
                lesson=lesson,
            )
            console.print(f"[green]Feedback recorded: {outcome}[/green]")
        else:
            console.print("[yellow]No task result to record[/yellow]")
        return True
    elif cmd == "/summarize":
        return "summarize"
    elif cmd == "/report":
        _handle_report_command(agent)
        return True
    elif cmd == "/help":
        console.print(render_help_text())
        return True
    elif cmd.startswith("/provider"):
        from dashscope.acli.cli.handlers_provider import (
            handle_provider_command,
        )

        handle_provider_command(cmd, agent, config)
        return True
    elif cmd.startswith("/dev test provider") or cmd.startswith(
        "/dev debug call",
    ):
        # Provider connectivity test and manual tool call must run async; the
        # synchronous dev handler would try to start a nested event loop and
        # crash inside the TUI.
        return "async"
    elif cmd.startswith("/dev"):
        handle_dev_command(cmd, config)
        return True
    elif cmd.startswith("/trust"):
        _handle_trust_command(cmd, agent)
        return True
    elif cmd.startswith("/rule"):
        _handle_rule_command(cmd, config)
        return True
    elif cmd.startswith("/skill"):
        return "skill"
    elif cmd.startswith("/mcp"):
        return "async"
    elif cmd.startswith("/profile"):
        return "async"
    elif cmd.startswith("/memory"):
        return "async"
    elif cmd.startswith("/session"):
        _handle_session_command(cmd, config, agent)
        return True
    elif cmd.startswith("/setup"):
        return "async"
    elif cmd.startswith("/capability"):
        return "async"
    elif cmd.startswith("/subagents"):
        from dashscope.acli.agents.subagents import handle_subagents_command

        handle_subagents_command(cmd, config)
        return True
    elif cmd.startswith("/cron"):
        return "async"
    elif cmd.startswith("/theme"):
        _handle_theme_command(cmd, config)
        return True
    elif cmd.startswith("/history"):
        _handle_history_command(cmd)
        return True
    elif cmd.startswith("/privacy"):
        _handle_privacy_command(cmd, config)
        return True
    elif cmd.startswith("/debug"):
        _handle_debug_command(cmd, config)
        return True
    elif cmd.startswith("/log"):
        _handle_log_command(cmd, agent)
        return True
    elif cmd.startswith("/trace"):
        _handle_trace_command(cmd, agent)
        return True
    elif cmd.startswith("/directives"):
        _handle_directives_command(cmd, config)
        return True
    elif cmd.startswith("/example"):
        from dashscope.acli.cli.examples import _handle_example_command

        args = cmd.split()[1:]
        _handle_example_command(args)
        return True
    elif cmd.startswith("/audit"):
        from dashscope.acli.audit import get_audit_logger

        logger = get_audit_logger()
        parts = cmd.strip().split(None, 1)
        sub = parts[1].strip() if len(parts) > 1 else "recent"
        sub_tokens = sub.split()
        action = sub_tokens[0] if sub_tokens else "recent"
        if action == "recent":
            limit = 20
            if len(sub_tokens) > 1:
                try:
                    limit = int(sub_tokens[1])
                    if limit <= 0:
                        raise ValueError
                except ValueError:
                    console.print(
                        "[yellow]Usage: /audit recent [N] "
                        "(N = positive integer)[/yellow]",
                    )
                    return True
            events = logger.recent(limit=limit)
        elif action == "clear":
            logger.clear()
            console.print("[green]✓ Audit log cleared[/green]")
            return True
        else:
            events = logger.query(limit=50)
        if not events:
            console.print("[dim]No audit records[/dim]")
            return True
        console.print(f"[bold]Audit log ({len(events)} entries)[/bold]")
        for ev in events:
            ts = ev.get("timestamp", "")[:19]
            src = ev.get("source", "")
            act = ev.get("action", "")
            subj = ev.get("subject", "")[:50]
            dec = ev.get("decision", "")
            console.print(
                f"  [dim]{ts}[/dim] [{src}] {act}: {subj} → "
                f"[yellow]{dec}[/yellow]",
            )
        return True
    return False


async def dispatch_async_command(
    cmd: str,
    config: Config,
    agent: Agent,
) -> None:
    """Dispatch async slash commands that need awaitable handlers.

    Centralizes the prefix-based routing previously duplicated between the
    CLI main loop and the TUI `_handle_async_command` method.
    """
    from dashscope.acli.cli.handlers_capability import (
        _handle_capability_command,
        _require_capability,
    )
    from dashscope.acli.cli.handlers_profile import (
        _handle_memory_command,
        _handle_profile_command,
    )
    from dashscope.acli.cli.handlers_setup import _handle_setup
    from dashscope.acli.cli.handlers_skill import _handle_cron_command
    from dashscope.acli.cli.mcp import _handle_mcp_command

    if cmd.startswith("/profile"):
        await _handle_profile_command(cmd, config)
    elif cmd.startswith("/memory"):
        await _handle_memory_command(cmd)
    elif cmd.startswith("/setup"):
        await _handle_setup(config, agent)
    elif cmd.startswith("/capability"):
        _handle_capability_command(cmd, config)
    elif cmd.startswith("/cron"):
        await _handle_cron_command(cmd, config, agent)
    elif cmd.startswith("/dev test provider"):
        parts = cmd.strip().split()
        if len(parts) >= 4:
            from dashscope.acli.dev import _test_provider

            await _test_provider(parts[3], config)
        else:
            # No provider name — let the sync handler print usage.
            handle_dev_command(cmd, config)
    elif cmd.startswith("/dev debug call"):
        from dashscope.acli.dev import _debug_call

        await _debug_call(cmd.strip().split())
    elif cmd.startswith("/dev"):
        handle_dev_command(cmd, config)
    else:
        if _require_capability(config, "bailian.mcp"):
            await _handle_mcp_command(cmd, config)


async def _handle_skill_continue(
    cmd: str,
    config: Config,
    agent=None,
) -> str | None:
    """Handle /skill: render the skill prompt if applicable, else return None.

    Used by both CLI and TUI so the `/skill` branch is not duplicated.
    """
    from dashscope.acli.cli.handlers_skill import _handle_skill_command

    skill_result = await _handle_skill_command(cmd, config, agent)
    return skill_result if skill_result else None
