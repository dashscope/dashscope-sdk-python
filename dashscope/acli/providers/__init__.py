# -*- coding: utf-8 -*-
from __future__ import annotations

from dashscope.acli.providers.base import LLMProvider, LLMResponse, ToolCall
from dashscope.acli.providers.profile import (
    ProviderChain,
    ProviderProfile,
    build_profiles_from_config,
    validate_key_for_profile,
)
from dashscope.acli.providers.tongyi import TongyiProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "TongyiProvider",
    "ProviderProfile",
    "ProviderChain",
    "build_profiles_from_config",
    "validate_key_for_profile",
    "get_provider",
    "get_provider_chain",
    "_create_provider",
    "PROVIDER_API_KEY_ENVS",
]

# Env var hints for the /key command and profile validation. NOT used for
# dispatch — protocol routing is driven by profile.protocol (filled from
# the user's custom-extensions.toml, see the basic-chat example).
PROVIDER_API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "tongyi": "DASHSCOPE_API_KEY",
}


def _create_provider(profile: ProviderProfile):
    """Instantiate a concrete provider from a profile.

    Dispatch priority:
      1. Custom extension provider (custom-extensions.toml) → use its
         resolved protocol
      2. Built-in provider name:
         - tongyi / dashscope → TongyiProvider (protocol controls
           message conversion)
         - anthropic          → AnthropicProvider
         - openai / other     → OpenAIProvider (default)

    base_url / api_key / protocol are filled by build_profiles_from_config
    (Config field wins) and refined here via find_provider (user
    custom-extensions.toml provides fallbacks for declared providers).
    """
    model = profile.model
    api_key = profile.api_key
    base_url = profile.base_url
    proto = (profile.protocol or "openai").lower()
    provider_name = (profile.provider or "").lower()

    from dashscope.acli.extensions import find_provider as _find_ext_provider

    ext = _find_ext_provider(profile.provider)
    if ext is not None:
        if not base_url:
            base_url = ext.base_url
        if not api_key:
            api_key = ext.resolve_api_key()
        if not (profile.protocol or "").strip():
            proto = ext.resolved_protocol()

    if provider_name in ("tongyi", "dashscope"):
        return TongyiProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            protocol=proto,
        )
    if proto == "anthropic" or provider_name == "anthropic":
        from dashscope.acli.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    if proto == "dashscope":
        return TongyiProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            protocol=proto,
        )
    from dashscope.acli.providers.openai import OpenAIProvider

    return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)


def get_provider(
    provider_name: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    protocol: str = "openai",
):
    """Backward-compatible single-provider factory."""
    profile = ProviderProfile(
        name=provider_name,
        provider=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        protocol=protocol,
    )
    return _create_provider(profile)


def get_provider_chain(config) -> ProviderChain:
    """Build a ProviderChain from acli Config (primary + fallbacks)."""
    profiles = build_profiles_from_config(config)
    if not profiles:
        raise RuntimeError("No usable provider configuration")

    # Surface key/url mismatches as warnings for the primary provider only.
    if profiles:
        warnings = validate_key_for_profile(profiles[0])
        if warnings:
            from rich.console import Console

            console = Console()
            for w in warnings:
                console.print(f"[yellow]⚠️  {w}[/yellow]")

    return ProviderChain(profiles)
