# -*- coding: utf-8 -*-
"""API key management command handlers."""
# pylint: disable=too-many-branches,unused-argument

from __future__ import annotations

import sys

from rich.console import Console

from dashscope.acli.cli.constants import KEY_TARGETS
from dashscope.acli.config import PROVIDER_MODELS, Config
from dashscope.acli.extensions import find_provider
from dashscope.acli.providers import get_provider_chain
from dashscope.acli.providers.profile import build_profiles_from_config

console = Console()


def all_key_targets(config: Config | None = None) -> dict[str, dict]:
    """Merge KEY_TARGETS (built-in) with extension providers into one dict.

    Extension provider entries are synthesized from CustomProvider fields so
    they look identical to built-in targets downstream.
    """
    merged: dict[str, dict] = dict(KEY_TARGETS)
    try:
        from dashscope.acli.extensions import current
    except Exception:
        return merged

    for prov in current().providers:
        if prov.name in merged:
            continue
        if not prov.auth:
            merged[prov.name] = {
                "field": f"{prov.name}_api_key",
                "env": "",
                "scope": "global",
                "desc": f"{prov.name} (no auth)",
                "no_auth": True,
            }
            continue
        merged[prov.name] = {
            "field": f"{prov.name}_api_key",
            "env": prov.api_key_env or "",
            "scope": "global",
            "desc": prov.name,
            "extension": True,
        }
    return merged


def _prompt_input(prompt: str, secret: bool = False) -> str:
    """Read user input. When secret=True, characters are not echoed."""
    try:
        if secret:
            import getpass

            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def ensure_provider_key(config: Config, agent) -> bool:
    """If the active provider has no resolvable key, prompt the user.

    Built-in providers persist to the global config slot; extension providers
    persist an encrypted token into custom-extensions.toml. Returns True when
    the key is available (or the user opts to set it later), False should not
    happen because missing/empty input exits the process.
    """
    if getattr(config, "_embedded_mode", False):
        return True

    profiles = build_profiles_from_config(config)
    if profiles and profiles[0].api_key:
        return True

    ext = find_provider(config.provider)
    targets = all_key_targets(config)
    key_info = targets.get(config.provider)
    if key_info:
        env_name = key_info.get("env") or ""
    elif ext is not None:
        env_name = ext.api_key_env or ""
    else:
        env_name = f"{config.provider.upper()}_API_KEY"

    console.print(f"\n[yellow]未检测到 {config.provider} 的 API Key[/yellow]")
    console.print("请选择设置方式:")
    if env_name:
        console.print(f"  [1] 设置环境变量 {env_name} (退出后设置)")
    else:
        console.print("  [1] 设置对应环境变量 (退出后设置)")
    console.print("  [2] 现在输入 API Key")
    console.print("  [3] 启动后使用 /provider 命令设置")
    choice = input("\n输入选项 [1/2/3]: ").strip()

    if choice == "1":
        console.print("[dim]请设置环境变量后重新启动:[/dim]")
        if env_name:
            console.print(f"[dim]  export {env_name}=sk-xxx[/dim]")
        sys.exit(0)
    elif choice == "2":
        api_key = _prompt_input("API Key: ", secret=True)
        if not api_key:
            console.print("[red]未输入 API Key，退出[/red]")
            sys.exit(0)
        if ext is not None:
            if _set_extension_provider_token(
                ext,
                config,
                direct_value=api_key,
            ):
                agent.provider = get_provider_chain(config)
                return True
            sys.exit(1)
        key_field = f"{config.provider}_api_key"
        if hasattr(config, key_field):
            setattr(config, key_field, api_key)
        else:
            config.api_key = api_key
        config.save_global()
        console.print("[green]API Key 已保存[/green]")
        agent.provider = get_provider_chain(config)
        return True
    elif choice == "3":
        console.print("[dim]提示: 启动后输入 /provider 设置 API Key[/dim]\n")
        return True
    else:
        console.print("[red]未设置 API Key，退出[/red]")
        sys.exit(0)


def _set_extension_cap_token(cap_key: str, direct_value: str = "") -> None:
    """Encrypt + persist a token into the [[capabilities]] block of
    custom-extensions.toml. If `direct_value` is empty, prompts (no echo).
    Picks the toml file the capability came from; defaults to global."""
    from pathlib import Path

    from dashscope.acli.extensions import (
        GLOBAL_EXTENSIONS_FILE,
        apply_extensions,
        auth_env_name,
        encrypt_for_toml,
        find_capability,
        set_capability_secret,
    )

    cap = find_capability(cap_key)
    if cap is None:
        console.print(f"[red]{cap_key} 不是已注册的扩展能力[/red]")
        return
    if (
        cap.auth
        and cap.auth.startswith("apikey-header:")
        or cap.auth.startswith("bearer:$")
        or cap.api_key_env
    ):
        env_hint = cap.api_key_env or auth_env_name(cap.auth)
    else:
        console.print(
            f"[yellow]{cap_key} 当前 auth 配置 "
            f"({cap.auth or 'none'}) 不需要 token[/yellow]",
        )
        return

    secret = direct_value or _prompt_input(
        f"  {cap_key} 凭证 (env {env_hint} 的值, 输入隐藏): ",
        secret=True,
    )
    if not secret:
        console.print("[dim]已取消[/dim]")
        return

    enc = encrypt_for_toml(secret)
    target = Path(cap.source) if cap.source else GLOBAL_EXTENSIONS_FILE
    if not set_capability_secret(target, cap_key, api_key_enc=enc):
        console.print(f"[red]写入失败: 在 {target} 中找不到 {cap_key} 块[/red]")
        return
    apply_extensions(PROVIDER_MODELS)
    console.print(f"[green]✓ 已加密保存到 {target}[/green]")
    console.print(f"[dim]运行时优先级: 环境变量 {env_hint} > 加密 token[/dim]")


def _set_extension_provider_token(
    ext_prov,
    config: Config,
    direct_value: str = "",
) -> bool:
    """Persist an extension provider API key as <name>_api_key in
    ~/.acli/config.toml, just like the built-in providers. If `direct_value`
    is empty, prompts (no echo).

    Returns True on success, False on cancellation or write failure
    (an error message is already printed on failure).
    """
    secret = direct_value or _prompt_input(
        f"  {ext_prov.name} API Key (输入隐藏，加密后写入): ",
        secret=True,
    )
    if not secret:
        console.print("[dim]已取消[/dim]")
        return False

    # Save to the provider's dynamic slot, e.g. ideatalk_api_key.
    old_provider = config.provider
    try:
        config.provider = ext_prov.name
        config.api_key = secret
    finally:
        config.provider = old_provider
    config.save_global()
    env_hint = ext_prov.api_key_env or "(无)"
    console.print(
        f"[green]✓ {ext_prov.name}_api_key 已加密保存到 ~/.acli/config.toml[/green]",
    )
    console.print(f"[dim]运行时优先级: 环境变量 {env_hint} > 加密 token[/dim]")
    return True


def _maybe_prompt_extension_token(cap_key: str, config=None) -> None:
    """Called from /capability enable. Detects when an extension capability
    needs a token (auth references env var) but neither env nor encrypted
    storage is providing one — and offers to capture it inline. Skips the
    prompt when a built-in provider already holds a key for the same env
    var (reused at registration time via runtime_key)."""
    import os

    from dashscope.acli.extensions import (
        auth_env_name,
        find_capability,
        provider_key_for_env,
    )

    cap = find_capability(cap_key)
    if cap is None:
        return  # built-in capability
    env_name = cap.api_key_env or auth_env_name(cap.auth)
    if not env_name:
        return  # no token needed (auth=none or unset)
    if os.environ.get(env_name) or cap.api_key_enc:
        return  # already covered
    if provider_key_for_env(config, env_name, KEY_TARGETS):
        # Same secret already stored for the matching built-in provider
        # (e.g. tongyi ↔ DASHSCOPE_API_KEY) — no second prompt needed.
        return
    console.print(
        f"[yellow]{cap_key} 缺凭证[/yellow] "
        f"[dim](env {env_name} 未设；toml 中也无加密 key)[/dim]",
    )
    yn = input("现在录入? (加密保存，下次自动用) [Y/n]: ").strip().lower()
    if yn in ("", "y", "yes"):
        _set_extension_cap_token(cap_key)
