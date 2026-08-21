# -*- coding: utf-8 -*-
"""Subagent management — discover, enable/disable, configure subagents.

Subagents are a subset of capabilities that function as autonomous workers
you delegate tasks to (vs. tool capabilities that the main agent calls).
Currently: local.subagent (built-in) + extension capabilities that act as
remote agents (e.g., Coze workflows).

The /subagents command provides:
  - list: show all discovered subagents with status
  - reload: re-scan custom-extensions.toml without restart
  - enable/disable: toggle subagents (delegates to capability system)
  - config: per-subagent settings (model, temperature, max_turns)
"""
# pylint: disable=wrong-import-position,too-many-return-statements
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dashscope.acli.config import Config

from rich.console import Console

console = Console()


# Capability keys that are subagents (vs. tool capabilities)
SUBAGENT_CAPABILITY_KEYS = {
    "local.subagent",
    "local.delegate",
}


@dataclass
class SubagentDef:
    """Discovered subagent definition."""

    key: str
    name: str
    description: str
    source: str  # "built-in", "extension", "remote"
    category: str  # "local", "remote"
    enabled: bool


def discover_subagents(config: Config) -> list[SubagentDef]:
    """Scan capabilities to find subagent-type ones."""
    from dashscope.acli.cli import CAPABILITY_CATALOG, _cap_enabled
    from dashscope.acli.extensions import current as ext_current

    ext = ext_current()
    ext_cap_keys = {c.key for c in ext.capabilities}

    subagents = []
    for cap in CAPABILITY_CATALOG:
        if cap["key"] not in SUBAGENT_CAPABILITY_KEYS:
            continue
        source = "extension" if cap["key"] in ext_cap_keys else "built-in"
        category = "local" if cap["platform"] == "local" else "remote"
        subagents.append(
            SubagentDef(
                key=cap["key"],
                name=cap["name"],
                description=f"{cap['platform']}.{cap['cap']}",
                source=source,
                category=category,
                enabled=_cap_enabled(config, cap["key"]),
            ),
        )
    return subagents


# ===== Per-subagent config get/set =====


def set_subagent_config(config: Config, key: str, **kwargs):
    """Create or update a subagent's config with the supplied overrides."""
    from dashscope.acli.config import SubagentConfig

    ac = config.subagents.get(key, SubagentConfig())
    for k, v in kwargs.items():
        if hasattr(ac, k):
            setattr(ac, k, v)
    config.subagents[key] = ac
    return ac


# ===== /subagents command dispatch =====


def handle_subagents_command(cmd: str, config: Config) -> None:
    """Handle /subagents <subcommand>."""
    parts = cmd.strip().split()
    sub = parts[1] if len(parts) > 1 else ""
    args = parts[2:]

    if not sub or sub == "list":
        _subagents_list(config)
    elif sub == "reload":
        _subagents_reload(config)
    elif sub == "enable":
        _subagents_enable(config, args)
    elif sub == "disable":
        _subagents_disable(config, args)
    elif sub == "config":
        _subagents_config(config, args)
    else:
        console.print(
            "[dim]Usage: /subagents [list|reload|enable <name>|disable "
            "<name>|config <name> [key value]][/dim]",
        )


def _subagents_list(config: Config) -> None:
    """List all discovered subagents with status."""
    subagents = discover_subagents(config)
    console.print("[bold]Available subagents:[/bold]")
    console.print("=" * 50)

    if not subagents:
        console.print("[dim]  (no subagents found)[/dim]")
        return

    for a in subagents:
        status = "[green]✓[/green]" if a.enabled else "[dim]✗[/dim]"
        cfg = config.subagents.get(a.key)
        cfg_str = ""
        if cfg:
            parts = []
            if cfg.model:
                parts.append(f"model={cfg.model}")
            if cfg.temperature:
                parts.append(f"temp={cfg.temperature}")
            if cfg.max_turns:
                parts.append(f"turns={cfg.max_turns}")
            if parts:
                cfg_str = f"  [cyan][{', '.join(parts)}][/cyan]"

        console.print(f"  {status} {a.key:25s} — {a.name}")
        console.print(
            f"     [dim]{a.description}  "
            f"[{a.source}/{a.category}]{cfg_str}[/dim]",
        )

    console.print(
        "\n[dim]Usage: /subagents enable|disable|config <name>[/dim]",
    )


def _subagents_reload(config: Config) -> None:
    # pylint: disable=unused-argument
    """Re-scan custom_extensions.toml and refresh subagent registry."""
    from dashscope.acli.cli import (
        PROVIDER_MODELS,
        sync_extensions_into_catalog,
    )
    from dashscope.acli.extensions import apply_extensions

    ext = apply_extensions(PROVIDER_MODELS)
    sync_extensions_into_catalog(ext)
    console.print("[green]✓ Subagent registry reloaded[/green]")


def _subagents_enable(config: Config, args: list[str]) -> None:
    """Enable a specific subagent."""
    if not args:
        console.print("[red]Missing subagent name[/red]")
        console.print("[dim]Usage: /subagents enable <name>[/dim]")
        return

    key = args[0]
    if key not in SUBAGENT_CAPABILITY_KEYS:
        console.print(f"[red]Unknown subagent: {key}[/red]")
        console.print(
            f"[dim]Options: "
            f"{', '.join(sorted(SUBAGENT_CAPABILITY_KEYS))}[/dim]",
        )
        return

    if config.enabled_capabilities is None:
        console.print(
            f"[dim]{key} already enabled (all-enabled state)[/dim]",
        )
        return

    if key in config.enabled_capabilities:
        console.print(f"[dim]{key} already enabled[/dim]")
        return

    config.enabled_capabilities.append(key)
    config.save_workspace()
    from dashscope.acli.tools.platform import register_one_capability

    added = register_one_capability(config, key)
    suffix = f" ({added} tools registered)" if added else ""
    console.print(f"[green]✓ Enabled: {key}[/green]{suffix}")


def _subagents_disable(config: Config, args: list[str]) -> None:
    """Disable a specific subagent."""
    if not args:
        console.print("[red]Missing subagent name[/red]")
        console.print("[dim]Usage: /subagents disable <name>[/dim]")
        return

    key = args[0]
    if key not in SUBAGENT_CAPABILITY_KEYS:
        console.print(f"[red]Unknown subagent: {key}[/red]")
        console.print(
            f"[dim]Options: "
            f"{', '.join(sorted(SUBAGENT_CAPABILITY_KEYS))}[/dim]",
        )
        return

    if config.enabled_capabilities is None:
        from dashscope.acli.cli import ALL_CAPABILITY_KEYS

        config.enabled_capabilities = [
            k for k in ALL_CAPABILITY_KEYS if k != key
        ]
    elif key in config.enabled_capabilities:
        config.enabled_capabilities.remove(key)
    else:
        console.print(f"[dim]{key} not enabled[/dim]")
        return

    config.save_workspace()
    from dashscope.acli.tools.platform import unregister_capability_tools

    removed = unregister_capability_tools(key)
    console.print(
        f"[yellow]✗ Disabled: {key}[/yellow] "
        f"({removed} tools unregistered)",
    )


def _subagents_config(config: Config, args: list[str]) -> None:
    """View or set per-subagent configuration.

    Syntax:
      /subagents config <name>                  — show current config
      /subagents config <name> <key> <value>    — set a config value

    Supported keys: model, temperature, max_turns
    """
    if not args:
        # Show all subagent configs
        console.print("[bold]Subagent configs:[/bold]")
        for key in SUBAGENT_CAPABILITY_KEYS:
            cfg = config.subagents.get(key)
            if cfg:
                console.print(f"  [cyan]{key}[/cyan]")
                console.print(f"    model: {cfg.model or '(default)'}")
                console.print(
                    f"    temperature: {cfg.temperature or '(default)'}",
                )
                console.print(
                    f"    max_turns: {cfg.max_turns or '(default)'}",
                )
            else:
                console.print(f"  [dim]{key}: (no custom config)[/dim]")
        return

    key = args[0]
    if key not in SUBAGENT_CAPABILITY_KEYS:
        console.print(f"[red]Unknown subagent: {key}[/red]")
        console.print(
            f"[dim]Options: "
            f"{', '.join(sorted(SUBAGENT_CAPABILITY_KEYS))}[/dim]",
        )
        return

    if len(args) == 1:
        # Show config for this subagent
        from dashscope.acli.config import SubagentConfig

        cfg = config.subagents.get(key, SubagentConfig())
        console.print(f"[bold]{key} config:[/bold]")
        console.print(f"  model: {cfg.model or '(default)'}")
        console.print(f"  temperature: {cfg.temperature or '(default)'}")
        console.print(f"  max_turns: {cfg.max_turns or '(default)'}")
        console.print(
            f"\n[dim]Usage: /subagents config {key} "
            f"<model|temperature|max_turns> <value>[/dim]",
        )
        return

    if len(args) < 3:
        console.print("[red]Missing value[/red]")
        console.print(
            f"[dim]Usage: /subagents config {key} "
            f"<model|temperature|max_turns> <value>[/dim]",
        )
        return

    cfg_key = args[1]
    cfg_val = args[2]

    if cfg_key == "model":
        set_subagent_config(config, key, model=cfg_val)
    elif cfg_key == "temperature":
        try:
            temp = float(cfg_val)
            if not 0.0 <= temp <= 2.0:
                console.print(
                    "[red]temperature must be between 0.0-2.0[/red]",
                )
                return
            set_subagent_config(config, key, temperature=temp)
        except ValueError:
            console.print("[red]temperature must be a number[/red]")
            return
    elif cfg_key == "max_turns":
        try:
            turns = int(cfg_val)
            if not 1 <= turns <= 100:
                console.print(
                    "[red]max_turns must be between 1-100[/red]",
                )
                return
            set_subagent_config(config, key, max_turns=turns)
        except ValueError:
            console.print("[red]max_turns must be an integer[/red]")
            return
    else:
        console.print(f"[red]Unknown config key: {cfg_key}[/red]")
        console.print("[dim]Options: model, temperature, max_turns[/dim]")
        return

    config.save_workspace()
    console.print(f"[green]✓ Updated {key}.{cfg_key} = {cfg_val}[/green]")
