# -*- coding: utf-8 -*-
"""Provider configuration wizard (/provider).

Q&A-style, modeled on /setup: one linear flow —
provider → key → model → protocol — with Enter accepting the default at
every step. The former arg-style subcommands (/provider use|model|protocol|
key) were removed — /provider always opens the wizard.
"""
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

from rich.console import Console

from dashscope.acli.agent import Agent
from dashscope.acli.cli.startup import _print_provider_debug
from dashscope.acli.config import (
    PROVIDER_MODELS,
    PROVIDERS,
    Config,
    normalize_model_name,
    register_custom_model,
)
from dashscope.acli.extensions import find_provider
from dashscope.acli.providers import get_provider_chain

console = Console()


def _read_choice(prompt: str, default: str = "") -> str:
    """Read a menu choice; EOF/Ctrl-C accepts the default."""
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        console.print()
        return default


def _read_secret(prompt: str) -> str:
    """Read a secret (no echo); EOF/Ctrl-C means empty."""
    try:
        import getpass

        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def _numbered_pick(
    label: str,
    options: list[str],
    default: str,
    custom_hint: str = "",
) -> str:
    """Show a numbered list and read a choice (number or name).

    Empty input keeps `default`. Free-form names are returned as-is so
    callers can accept unlisted values (e.g. custom model names).
    """
    if options:
        console.print(f"[bold]{label}:[/bold]")
        for i, name in enumerate(options, 1):
            marker = "[green]✓[/green]" if name == default else " "
            console.print(f"  [{i}] {marker} {name}")
        if custom_hint:
            console.print(f"[dim]{custom_hint}[/dim]")
    hint = f" [{default}]" if default else ""
    raw = _read_choice(f"Choose (number or name){hint}: ", default=default)
    if raw.isdigit() and options and 1 <= int(raw) <= len(options):
        raw = options[int(raw) - 1]
    return raw


def _rebuild_provider(agent: Agent, config: Config) -> bool:
    """Rebuild the provider chain after a config change; sync agent names."""
    try:
        agent.provider = get_provider_chain(config)
    except Exception as e:
        console.print(f"[red]Switch failed: {e}[/red]")
        console.print(
            "[dim]If an API Key is missing, run /provider again[/dim]",
        )
        return False
    agent.provider_name = config.provider
    agent.model_name = config.model
    _print_provider_debug(agent.provider)
    return True


def _warn_missing_ext_key(config: Config) -> None:
    """Warn when the active extension provider has no resolvable key —
    otherwise the chain would silently fall back on auth failure."""
    ext = find_provider(config.provider)
    if (
        ext is None
        or config.provider in PROVIDERS
        or ext.resolve_api_key(config)
    ):
        return
    env_hint = ext.api_key_env or "(none)"
    console.print(
        f"[yellow]⚠️  {config.provider} is missing an API Key "
        f"(env {env_hint} unset; no encrypted key in toml either)"
        f"[/yellow]",
    )


def _wizard_key_step(config: Config) -> None:
    """Step 2: API key — always shown; Enter keeps the current key / skips."""
    from dashscope.acli.cli.handlers_key import (
        _set_extension_provider_token,
        all_key_targets,
    )

    info = all_key_targets(config).get(config.provider)
    if info is None:
        return
    if info.get("no_auth"):
        console.print(
            f"[dim]{config.provider} needs no API Key (auth = false)[/dim]",
        )
        return

    existing = getattr(config, info["field"], "") or ""
    if not existing and info.get("extension"):
        # May still resolve from the toml-encrypted key or an env var.
        ext = find_provider(config.provider)
        if ext is not None:
            existing = ext.resolve_api_key(config) or ""

    if existing:
        hint = f" [set …{existing[-4:]}, Enter to keep]"
    elif info.get("env"):
        hint = f" ({info['env']}) [Enter to skip]"
    else:
        hint = " [Enter to skip]"
    value = _read_secret(f"\n{config.provider} API Key{hint}: ")
    if not value:
        if not existing:
            env_hint = f"or env var {info['env']} " if info.get("env") else ""
            console.print(
                f"[dim]Skipped {config.provider} key; set it later "
                f"via /provider {env_hint}[/dim]",
            )
        return

    if info.get("extension"):
        ext = find_provider(config.provider)
        if ext is not None:
            _set_extension_provider_token(ext, config, direct_value=value)
        return
    setattr(config, info["field"], value)
    config.save_global()
    console.print(f"[green]✓ {config.provider} API Key saved[/green]")


def _provider_wizard(agent: Agent, config: Config) -> bool:
    """Bare /provider — linear Q&A: provider → key → model → protocol."""
    from dashscope.acli.extensions import current as ext_current

    console.print(
        f"[dim]Current: {config.provider}/{config.model}  "
        f"protocol={config.protocol}[/dim]\n",
    )

    loaded = ext_current()
    for err in loaded.errors:
        console.print(f"[yellow]custom-extensions.toml: {err}[/yellow]")

    # 1) Provider — Enter keeps the current one.
    names = list(PROVIDER_MODELS) + [
        p.name for p in loaded.providers if p.name not in PROVIDER_MODELS
    ]
    provider = _numbered_pick("Available providers", names, config.provider)
    if provider not in names:
        console.print(f"[red]Unknown provider: {provider}; cancelled[/red]")
        return True

    ext = find_provider(provider)
    if provider != config.provider:
        config.provider = provider
        config.base_url = ""  # old provider's endpoint override doesn't apply
        if ext is not None:
            ext_models = ext.resolved_models()
            config.model = ext.default_model or (
                ext_models[0] if ext_models else config.model
            )
            config.protocol = ext.resolved_protocol()
        elif PROVIDER_MODELS.get(provider):
            config.model = PROVIDER_MODELS[provider][0]

    # 2) API Key — always shown; Enter keeps the current key / skips.
    _wizard_key_step(config)
    _warn_missing_ext_key(config)

    # 3) Model — Enter keeps the provider default / current model.
    models = PROVIDER_MODELS.get(config.provider, [])
    if not models and ext is not None:
        models = ext.resolved_models()
    if models:
        default_model = config.model if config.model in models else models[0]
    else:
        default_model = config.model
        console.print(
            f"[dim]{config.provider} has no preset model list; "
            f"enter a model name directly[/dim]",
        )
    model = _numbered_pick(
        f"{config.provider} available models",
        models,
        default_model,
        custom_hint=(
            "Not listed? Enter a new model name to register it "
            "(same as /dev model add)"
        ),
    )
    if model:
        normalized = normalize_model_name(model)
        if normalized != model.strip():
            console.print(
                f"[dim]Model name normalized to {normalized} "
                f"(API model IDs are case-sensitive)[/dim]",
            )
        if (
            models
            and normalized not in models
            and config.provider in PROVIDER_MODELS
        ):
            register_custom_model(config, config.provider, normalized)
            console.print(
                f"[green]✓ Registered {config.provider}/{normalized}[/green]",
            )
        config.model = normalized

    # 4) Protocol — for extension providers the default comes from the toml.
    ext_proto = ext.resolved_protocol() if ext is not None else None
    if ext_proto:
        console.print(
            f"[dim]{config.provider} toml declares protocol: "
            f"{ext_proto}[/dim]",
        )
    proto = _numbered_pick(
        "Protocol",
        ["openai", "anthropic"],
        config.protocol,
    )
    if proto in ("openai", "anthropic"):
        if ext_proto and proto != ext_proto:
            console.print(
                f"[yellow]⚠️  {config.provider}'s toml declares "
                f"protocol {ext_proto}; switching to {proto} may fail "
                f"with 404[/yellow]",
            )
        config.protocol = proto
    elif proto:
        console.print(
            f"[yellow]Unknown protocol '{proto}'; "
            f"keeping {config.protocol}[/yellow]",
        )

    config.save_global()
    config.save_workspace()
    if _rebuild_provider(agent, config):
        console.print(
            f"[green]✓ Switched: {config.provider}/{config.model}  "
            f"protocol={config.protocol}[/green]",
        )
    return True


def handle_provider_command(cmd: str, agent: Agent, config: Config) -> bool:
    """/provider always opens the interactive wizard.

    Trailing args from the retired subcommand form
    (/provider use|model|protocol|key ...) are ignored.
    """
    if cmd.split()[1:]:
        console.print(
            "[dim]/provider is now a Q&A wizard; args ignored[/dim]",
        )
    return _provider_wizard(agent, config)
