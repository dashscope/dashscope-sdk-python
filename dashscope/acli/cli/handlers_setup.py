# -*- coding: utf-8 -*-
"""Workspace setup command handlers."""

from __future__ import annotations

import re

from rich.console import Console

from dashscope.acli.cli.constants import (
    ALL_CAPABILITY_KEYS,
    CAPABILITY_CATALOG,
    KEY_TARGETS,
)
from dashscope.acli.config import PROVIDER_MODELS, Config, normalize_model_name
from dashscope.acli.utils.ids import stable_memory_user_id

console = Console()


def _prompt_input(prompt: str, secret: bool = False) -> str:
    """Read user input. When secret=True, characters are not echoed."""
    try:
        if secret:
            import getpass

            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _setup_user_name(config: Config) -> None:
    current_name = config.user_name or ""
    name = input(
        f"Username{f' [{current_name}]' if current_name else ''}: ",
    ).strip()
    if name:
        config.user_name = name
    elif not current_name:
        config.user_name = "anonymous"


def _ensure_provider_key(config: Config) -> None:
    """Prompt for the current provider's API key only if it's missing.

    Empty input means "skip for now"; the user can set it later via
    /provider or the corresponding environment variable.
    """
    key_info = KEY_TARGETS.get(config.provider)
    if not key_info:
        from dashscope.acli.cli.handlers_key import all_key_targets

        targets = all_key_targets(config)
        key_info = targets.get(config.provider)
    if not key_info:
        return
    field = key_info["field"]
    if getattr(config, field, ""):
        return
    value = _prompt_input(
        f"\n{config.provider} API Key ({key_info['env']}) [Enter to skip]: ",
        secret=True,
    )
    if value:
        setattr(config, field, value)
    else:
        env_hint = (
            f"or env var {key_info['env']} " if key_info.get("env") else ""
        )
        console.print(
            f"[dim]Skipped {config.provider} key; set it later via "
            f"/provider {env_hint}[/dim]",
        )


def _warn_unsatisfiable_capabilities(config: Config) -> None:
    """For each enabled capability with missing required config, prompt
    for each field — but treat empty input as 'defer'. The capability
    stays in enabled_capabilities regardless; what's missing simply
    means register_one_capability registers 0 tools at boot until the
    user fills the creds later (via /provider or re-enable).

    User intent: keep asking (so it's easy to fill in one go), but make
    'skip with Enter' frictionless and non-destructive.
    """
    from dashscope.acli.utils.sanitizer import is_secret_field

    caps_to_check = (
        config.enabled_capabilities
        if config.enabled_capabilities is not None
        else ALL_CAPABILITY_KEYS
    )
    deferred: list[tuple[str, list[str]]] = []
    for cap_info in CAPABILITY_CATALOG:
        if cap_info["key"] not in caps_to_check:
            continue
        missing_before = [
            f for f in cap_info["requires"] if not getattr(config, f, "")
        ]
        if not missing_before:
            continue
        console.print(
            f"\n[yellow]{cap_info['key']} needs configuration[/yellow] "
            f"[dim](Enter skips a field; the capability stays enabled "
            f"and can be completed later via /setup)[/dim]:",
        )
        for field in cap_info["requires"]:
            if getattr(config, field, ""):
                continue  # already set
            value = _prompt_input(
                f"  {field}: ",
                secret=is_secret_field(field),
            )
            if value:
                setattr(config, field, value)
        missing_after = [
            f for f in cap_info["requires"] if not getattr(config, f, "")
        ]
        if missing_after:
            deferred.append((cap_info["key"], missing_after))

    if deferred:
        console.print(
            "\n[yellow]These capabilities lack credentials; they stay "
            "enabled but are not callable yet:[/yellow]",
        )
        for cap_key, miss in deferred:
            console.print(
                f"  [dim]·[/dim] [bold]{cap_key}[/bold] "
                f"missing: {', '.join(miss)}",
            )
        console.print(
            "[dim]They take effect on the next launch once filled in; "
            "or run /setup, then [bold]/capability enable <cap>[/bold] "
            "to activate immediately.[/dim]",
        )


def _setup_finalize(config: Config, agent) -> None:
    """Common tail: derive memory_user_id, persist, rebuild agent provider,
    re-register tools based on new capabilities, print summary."""
    if not config.memory_user_id:
        config.memory_user_id = stable_memory_user_id()

    config.save_global()
    config.save_workspace()

    from dashscope.acli.providers import get_provider_chain

    agent.provider = get_provider_chain(config)
    agent.provider_name = config.provider
    agent.model_name = config.model
    agent.user_name = config.user_name

    # Re-register platform tools based on new enabled_capabilities
    from dashscope.acli.tools.platform import (
        _capability_tools,
        register_platform_tools,
        unregister_capability_tools,
    )

    # Clear all existing capability tools first
    for cap_key in list(_capability_tools.keys()):
        unregister_capability_tools(cap_key)
    # Then register based on new config (same wiring as boot: repl.py passes
    # connect_mcp_fn so the mcp_connect tool gets registered).
    from dashscope.acli.cli.mcp import _connect_mcp

    register_platform_tools(config, connect_mcp_fn=_connect_mcp)

    console.print("\n[green]✓ Config saved to .acli/config.toml[/green]")
    caps_display = (
        ALL_CAPABILITY_KEYS
        if config.enabled_capabilities is None
        else config.enabled_capabilities
    )
    from dashscope.acli.tools.registry import registry

    tool_count = len(registry.list_tools())
    console.print(f"[dim]  User: {config.user_name}[/dim]")
    console.print(f"[dim]  Model: {config.provider}/{config.model}[/dim]")
    console.print(
        f"[dim]  Capabilities: "
        f"{', '.join(caps_display) if caps_display else '(none)'}[/dim]",
    )
    console.print(f"[dim]  Tools: {tool_count} registered[/dim]")


# ---- preset 1: Alibaba/Bailian ----


def _apply_preset_bailian(config: Config) -> None:
    config.provider = "tongyi"
    config.model = "qwen3.7-plus"
    config.enabled_capabilities = [
        "bailian.mcp",
        "bailian.cli",
    ]
    _ensure_provider_key(config)
    _warn_unsatisfiable_capabilities(config)


# ---- preset 2: China general ----


def _apply_preset_china_common(config: Config) -> None:
    config.provider = "tongyi"
    config.model = "qwen-max"
    config.enabled_capabilities = ["bailian.mcp"]
    _ensure_provider_key(config)
    _warn_unsatisfiable_capabilities(config)


# ---- preset 3: custom (former _handle_setup full interactive flow) ----


async def _setup_preset_custom(config: Config) -> None:
    # username already collected by _handle_setup before preset dispatch
    from dashscope.acli.extensions import current

    ext = current()
    ext_providers = {p.name: p.resolved_models() for p in ext.providers}
    all_providers = list(PROVIDER_MODELS.keys()) + [
        p for p in ext_providers if p not in PROVIDER_MODELS
    ]
    console.print(f"\nAvailable providers: {', '.join(all_providers)}")
    provider_input = input(f"Provider [{config.provider}]: ").strip()
    if provider_input and provider_input in all_providers:
        config.provider = provider_input

    models = PROVIDER_MODELS.get(config.provider, []) or ext_providers.get(
        config.provider,
        [],
    )
    if models:
        console.print(f"Available models: {', '.join(models)}")
        # If current model isn't in this provider's list, suggest the first one
        default_model = config.model if config.model in models else models[0]
    else:
        default_model = config.model
    model_input = input(f"Model [{default_model}]: ").strip()
    if model_input:
        config.model = normalize_model_name(model_input)
    elif default_model != config.model:
        config.model = default_model

    _ensure_provider_key(config)

    console.print("\n[bold]Available platform capabilities:[/bold]")
    for i, cap in enumerate(CAPABILITY_CATALOG, 1):
        console.print(f"  [{i}] {cap['key']:20s} — {cap['name']}")

    console.print(
        "\n[bold]Enter the numbers of capabilities to enable[/bold] "
        "[dim](separate multiple with commas or spaces; "
        "Enter enables none)[/dim]",
    )

    selection = input("> ").strip()
    if selection:
        selected = []
        for part in re.split(r"[,\s]+", selection):
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(CAPABILITY_CATALOG):
                    selected.append(CAPABILITY_CATALOG[idx]["key"])
        config.enabled_capabilities = selected
    else:
        config.enabled_capabilities = []

    _warn_unsatisfiable_capabilities(config)


async def _handle_setup(config: Config, agent) -> None:
    """Workspace setup — ask username first, then pick a preset
    (default [1] Alibaba/Bailian), or take the full interactive flow."""
    console.print("[bold]Workspace setup[/bold]\n")

    # Username comes first — it's identity, not configuration. The preset
    # then says "for user X, here are sensible defaults".
    _setup_user_name(config)

    console.print("\nSelect a configuration mode:")
    console.print(
        "  [cyan][1][/cyan] [bold]Alibaba/Bailian[/bold] (default) — "
        "tongyi/qwen3.7-plus + bailian.mcp/cli",
    )
    console.print(
        "  [cyan][2][/cyan] China general        — "
        "tongyi/qwen-max + bailian.mcp",
    )
    console.print(
        "  [cyan][3][/cyan] Custom               — "
        "pick each item (Provider / Model / Capabilities)",
    )

    choice = input("\nChoice [1]: ").strip() or "1"
    if choice == "2":
        _apply_preset_china_common(config)
    elif choice == "3":
        await _setup_preset_custom(config)
    else:
        if choice != "1":
            console.print(
                f"[dim]Unrecognized '{choice}'; "
                f"using [1] Alibaba/Bailian[/dim]",
            )
        _apply_preset_bailian(config)

    _setup_finalize(config, agent)
