# -*- coding: utf-8 -*-
"""Capability management command handlers."""
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements

from __future__ import annotations

from rich.console import Console

from dashscope.acli.cli.constants import (
    ALL_CAPABILITY_KEYS,
    CAPABILITY_CATALOG,
)
from dashscope.acli.config import PROVIDER_MODELS, Config

console = Console()


def sync_extensions_into_catalog(ext) -> None:
    """Fold extension capabilities from custom-extensions.toml into
    CAPABILITY_CATALOG / ALL_CAPABILITY_KEYS so /capability list / enable
    / disable see them as first-class entries alongside built-ins.

    Called from _run_loop at startup AND from dev._hot_reload after any
    /dev capability add|remove — so newly added caps are immediately
    eligible for /capability enable without restarting.

    Idempotent + removal-aware: caps not in `ext.capabilities` but
    currently in the catalog as extension entries get dropped (covers
    /dev capability remove).
    """
    from dashscope.acli.tools.platform import _CAPABILITY_KEYS as _BUILTIN_KEYS
    from dashscope.acli.tools.platform import unregister_capability_tools

    ext_keys = {c.key for c in ext.capabilities}

    # Remove extension entries that no longer exist
    for i in range(len(CAPABILITY_CATALOG) - 1, -1, -1):
        cap = CAPABILITY_CATALOG[i]
        if cap["key"] in _BUILTIN_KEYS:
            continue  # built-ins stay
        if cap["key"] not in ext_keys:
            removed = CAPABILITY_CATALOG.pop(i)
            if removed["key"] in ALL_CAPABILITY_KEYS:
                ALL_CAPABILITY_KEYS.remove(removed["key"])
            # Cap vanished from toml (/dev capability rm): drop its tools too,
            # otherwise they stay registered until restart or manual disable.
            unregister_capability_tools(removed["key"])

    # Add new ones
    for cap in ext.capabilities:
        if cap.key in ALL_CAPABILITY_KEYS:
            continue
        platform_part, _, cap_part = cap.key.partition(".")
        CAPABILITY_CATALOG.append(
            {
                "key": cap.key,
                "name": cap.display or cap.key,
                "platform": platform_part,
                "cap": cap_part or cap.key,
                # creds checked at HTTP call time, not registration
                "requires": [],
            },
        )
        ALL_CAPABILITY_KEYS.append(cap.key)


def _cap_enabled(config: Config, cap_key: str) -> bool:
    """Check if a platform capability is enabled.

    None = all enabled (not configured).
    When privacy_mode is True, all cloud capabilities are disabled."""
    if config.privacy_mode:
        return False
    if config.enabled_capabilities is None:
        return True
    return cap_key in config.enabled_capabilities


def _require_capability(config: Config, cap_key: str) -> bool:
    """Gate a slash command on a platform capability.

    Returns True when the capability is enabled (or all enabled). Otherwise
    prints a hint with the available ways to enable it and returns False.
    """
    if _cap_enabled(config, cap_key):
        return True
    cap_name = next(
        (c["name"] for c in CAPABILITY_CATALOG if c["key"] == cap_key),
        cap_key,
    )
    console.print(
        f"[yellow]Capability not enabled: {cap_key} "
        f"({cap_name})[/yellow]\n"
        f"[dim]Ways to enable:[/dim]\n"
        f"[dim]  /capability enable {cap_key}    — enable now, "
        f"prompting for credentials as needed[/dim]\n"
        f"[dim]  /setup                          — re-select "
        f"capabilities[/dim]\n"
        f"[dim]  /capability list                  — view all "
        f"capability states[/dim]",
    )
    return False


def _prompt_missing_config(config: Config, requires: list[str]):
    """Prompt user for missing config fields, return True if all filled."""
    import getpass

    from dashscope.acli.utils.sanitizer import is_secret_field

    for field_name in requires:
        current = getattr(config, field_name, "")
        if not current:
            try:
                if is_secret_field(field_name):
                    value = getpass.getpass(f"  Enter {field_name}: ").strip()
                else:
                    value = input(f"  Enter {field_name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                value = ""
            if not value:
                return False
            setattr(config, field_name, value)
    return True


def _handle_capability_command(cmd: str, config: Config):
    """Handle /capability command — list/enable/disable capabilities."""
    parts = cmd.strip().split()
    sub = parts[1] if len(parts) > 1 else ""

    if not sub or sub == "list":
        console.print("[bold]Capability status:[/bold]")
        for cap in CAPABILITY_CATALOG:
            enabled = _cap_enabled(config, cap["key"])
            status = "[green]✓[/green]" if enabled else "[dim]✗[/dim]"
            console.print(f"  {status} {cap['key']:20s} — {cap['name']}")
        return

    if sub == "enable" and len(parts) > 2:
        cap_key = parts[2]
        if cap_key not in ALL_CAPABILITY_KEYS:
            console.print(f"[red]Unknown capability: {cap_key}[/red]")
            console.print(
                f"[dim]Options: {', '.join(ALL_CAPABILITY_KEYS)}[/dim]",
            )
            return
        if config.enabled_capabilities is None:
            console.print("[dim]All capabilities currently enabled[/dim]")
            return
        if cap_key in config.enabled_capabilities:
            # Already enabled — but extension caps may still lack credentials
            # (enabled without a token, or env var unset). Offer the prompt
            # again and re-register so the tools bind to fresh creds.
            from dashscope.acli.cli.handlers_key import (
                _maybe_prompt_extension_token,
            )

            _maybe_prompt_extension_token(cap_key, config)
            from dashscope.acli.cli.mcp import _connect_mcp
            from dashscope.acli.tools.platform import (
                register_one_capability,
                unregister_capability_tools,
            )

            # Drop stale registrations first so extension tool closures rebind
            # to the (possibly just-entered) credentials.
            unregister_capability_tools(cap_key)
            added = register_one_capability(
                config,
                cap_key,
                connect_mcp_fn=_connect_mcp,
            )
            suffix = f" ({added} tools registered)" if added else ""
            console.print(f"[dim]{cap_key} already enabled[/dim]{suffix}")
            return
        cap_info = next(c for c in CAPABILITY_CATALOG if c["key"] == cap_key)
        missing = [
            f for f in cap_info["requires"] if not getattr(config, f, "")
        ]
        if missing:
            console.print(
                f"[yellow]{cap_info['key']} needs " f"configuration:[/yellow]",
            )
            if not _prompt_missing_config(config, cap_info["requires"]):
                console.print("[red]Config incomplete; not enabled[/red]")
                return

        # Extension-capability bearer/apikey-header token
        from dashscope.acli.cli.handlers_key import (
            _maybe_prompt_extension_token,
        )

        _maybe_prompt_extension_token(cap_key, config)

        config.enabled_capabilities.append(cap_key)
        config.save_workspace()
        from dashscope.acli.cli.mcp import _connect_mcp
        from dashscope.acli.tools.platform import register_one_capability

        added = register_one_capability(
            config,
            cap_key,
            connect_mcp_fn=_connect_mcp,
        )
        suffix = (
            f" ({added} tools registered)"
            if added
            else " (no tools registered — credentials may be missing)"
        )
        console.print(f"[green]✓ Enabled: {cap_key}[/green]{suffix}")
        return

    if sub == "disable" and len(parts) > 2:
        cap_key = parts[2]
        if cap_key not in ALL_CAPABILITY_KEYS:
            console.print(f"[red]Unknown capability: {cap_key}[/red]")
            console.print(
                f"[dim]Options: {', '.join(ALL_CAPABILITY_KEYS)}[/dim]",
            )
            return
        if config.enabled_capabilities is None:
            config.enabled_capabilities = [
                k for k in ALL_CAPABILITY_KEYS if k != cap_key
            ]
        elif cap_key in config.enabled_capabilities:
            config.enabled_capabilities.remove(cap_key)
        else:
            console.print(f"[dim]{cap_key} not enabled[/dim]")
            return
        config.save_workspace()
        from dashscope.acli.tools.platform import unregister_capability_tools

        removed = unregister_capability_tools(cap_key)
        console.print(
            f"[yellow]✗ Disabled: {cap_key}[/yellow] "
            f"({removed} tools unregistered)",
        )
        return

    if sub in ("reload", "refresh"):
        from dashscope.acli.extensions import apply_extensions
        from dashscope.acli.tools.platform import (
            refresh_extension_capability_tools,
        )

        ext = apply_extensions(PROVIDER_MODELS)
        sync_extensions_into_catalog(ext)
        # Re-register enabled extension caps so tool closures rebind to the
        # freshly loaded toml (endpoint/auth/token edits take effect now).
        refreshed = refresh_extension_capability_tools(config)
        console.print(
            f"[green]✓ Capability registry reloaded[/green] "
            f"({refreshed} extension tools refreshed)",
        )
        return

    if sub == "config":
        args = parts[2:]
        if not args:
            console.print("[bold]Capability config:[/bold]")
            console.print(
                "[dim]Use /subagents config <name> <key> <value>[/dim]",
            )
            console.print(
                "[dim]Supported: " + ", ".join(ALL_CAPABILITY_KEYS) + "[/dim]",
            )
            return
        # Delegate to subagents config for subagent-type capabilities
        from dashscope.acli.agents.subagents import (
            SUBAGENT_CAPABILITY_KEYS,
            handle_subagents_command,
        )

        cap_key = args[0]
        if cap_key in SUBAGENT_CAPABILITY_KEYS:
            # Rewrite as /subagents config <rest...>
            handle_subagents_command(
                f"/subagents config {' '.join(args)}",
                config,
            )
        else:
            console.print(
                f"[dim]{cap_key} is not a configurable subagent; "
                f"use /subagents config[/dim]",
            )
        return

    # User typed a known cap without an action — suggest enable/disable for it
    if sub in ALL_CAPABILITY_KEYS:
        currently_enabled = _cap_enabled(config, sub)
        action_hint = "disable" if currently_enabled else "enable"
        console.print(
            f"[yellow]Missing action: enable or disable[/yellow]\n"
            f"[dim]Current {sub}: "
            f"{'enabled' if currently_enabled else 'disabled'}[/dim]\n"
            f"[dim]e.g.: /capability {action_hint} {sub}[/dim]",
        )
        return

    console.print(
        r"[dim]Usage: /capability \[list|enable <cap>|disable <cap>|"
        r"reload|config <cap>][/dim]" + "\n"
        f"[dim]Available caps: {', '.join(ALL_CAPABILITY_KEYS)}[/dim]",
    )
