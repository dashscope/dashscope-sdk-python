# -*- coding: utf-8 -*-
"""Provider profile registry and fallback chain.

A ProviderProfile captures everything needed to instantiate one LLM backend
(model, API key, base URL, timeout).  A ProviderChain holds an ordered list of
profiles and routes to the next one when the primary fails with a retryable
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from dashscope.acli.providers.base import LLMChunk, LLMProvider, LLMResponse
from dashscope.acli.providers.hardening import (
    HardenedProvider,
    is_retryable_error,
)

_API_KEY_ENVS = {
    "tongyi": "DASHSCOPE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Provider-specific API key prefixes used for key-to-URL sanity checks.
_KEY_PREFIXES = {
    "tongyi": ("sk-",),
    "anthropic": ("sk-ant",),
    "openai": ("sk-",),
}

# Known host fragments that imply a specific provider type.
_HOST_PROVIDER_HINTS = {
    "tongyi": ("dashscope.aliyuncs.com",),
    "anthropic": ("anthropic.com", "api.claude.ai"),
    "openai": ("openai.com", "api.openai.com"),
}

DEFAULT_BASE_URLS = {
    "tongyi": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
}


@dataclass
class ProviderProfile:
    """Everything needed to talk to one model endpoint."""

    name: str
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    timeout: float = 120.0
    protocol: str = "openai"
    max_retries: int = 2


def _host_of(url: str | None) -> str:
    if not url:
        return ""
    # Very small URL parser; works for http(s) and bare host:port.
    url = url.lower()
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split("/", 1)[0]


def _detect_provider_from_host(base_url: str | None) -> str | None:
    host = _host_of(base_url)
    for provider, hints in _HOST_PROVIDER_HINTS.items():
        if any(h in host for h in hints):
            return provider
    return None


def validate_key_for_profile(profile: ProviderProfile) -> list[str]:
    """Return warnings if the API key looks mismatched for the endpoint."""
    warnings: list[str] = []
    key = profile.api_key or ""
    base_url = profile.base_url or ""

    if not key:
        warnings.append(f"{profile.name}: API key is empty")
        return warnings

    # Detect endpoint type from host.
    detected = _detect_provider_from_host(base_url)
    if detected and detected != profile.provider:
        warnings.append(
            f"{profile.name}: provider='{profile.provider}' but base_url "
            f"'{base_url}' looks like '{detected}'",
        )

    # Check key prefix when the provider has a known convention.
    prefixes = _KEY_PREFIXES.get(profile.provider, ("",))
    if prefixes and prefixes != ("",):
        if not any(key.startswith(p) for p in prefixes):
            warnings.append(
                f"{profile.name}: API key does not start with expected "
                f"prefix {prefixes} for provider '{profile.provider}'",
            )

    return warnings


def build_profiles_from_config(config) -> list[ProviderProfile]:
    """Build primary + fallback provider profiles from acli Config."""
    profiles: list[ProviderProfile] = []

    def _profile_for(
        provider_name: str,
        model: str | None = None,
    ) -> ProviderProfile | None:
        api_key_attr = f"{provider_name}_api_key"
        api_key = getattr(config, api_key_attr, "") or ""

        # Resolve base_url: config.base_url only makes sense for the primary
        # provider; fallbacks use their well-known endpoints unless overridden.
        base_url = (
            config.base_url if provider_name == config.provider else None
        )

        timeout = float(getattr(config, "timeout", 120))
        # config.protocol describes the PRIMARY provider only; fallback
        # profiles must use their own extension-toml protocol (resolved
        # below) or the default, never inherit the primary's.
        if provider_name == config.provider:
            protocol = getattr(config, "protocol", "openai")
        else:
            protocol = ""

        # Consult the extension catalog (user custom-extensions.toml
        # at ~/.acli/ or ./.acli/) to fill base_url / protocol / api_key
        # when Config doesn't. Config field wins for api_key and protocol;
        # toml provides base_url + protocol fallbacks.
        try:
            from dashscope.acli.extensions import find_provider

            ext = find_provider(provider_name)
            if ext is not None:
                if not api_key:
                    api_key = ext.resolve_api_key(config) or ""
                if not base_url:
                    base_url = ext.base_url
                if not protocol:
                    protocol = ext.resolved_protocol()
        except Exception:
            pass

        if not protocol:
            protocol = "openai"

        if not base_url:
            base_url = DEFAULT_BASE_URLS.get(provider_name, "")

        return ProviderProfile(
            name=provider_name,
            provider=provider_name,
            model=model or _default_model_for(provider_name, config),
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            protocol=protocol,
        )

    # Primary profile.
    primary = _profile_for(config.provider, config.model)
    if primary:
        profiles.append(primary)

    # Fallback profiles.
    fallback_names = getattr(config, "fallback_providers", []) or []
    if not fallback_names:
        # If no explicit fallback list, try a sensible default based on
        # env keys.
        fallback_names = _infer_fallback_names(config)

    seen = {config.provider}
    for name in fallback_names:
        if name in seen:
            continue
        seen.add(name)
        prof = _profile_for(name)
        if not prof:
            continue
        # Skip only if the provider genuinely requires a key but has none.
        # Keyless providers (ollama, local proxies with auth=False) must not
        # be excluded here — they have an empty api_key by design.
        needs_key = (
            _API_KEY_ENVS.get(name) is not None
        )  # None sentinel = no key required
        if needs_key and not prof.api_key:
            continue
        profiles.append(prof)

    return profiles


def _default_model_for(provider: str, config) -> str:
    from dashscope.acli.config import PROVIDER_MODELS

    models = PROVIDER_MODELS.get(provider, [])
    if models:
        return models[0]
    # Extension provider: use config model as a guess.
    return getattr(config, "model", "")


def _infer_fallback_names(config) -> list[str]:
    """Infer fallback providers from available API keys when not configured."""
    import os

    candidates = []
    if config.provider != "anthropic" and (
        config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    ):
        candidates.append("anthropic")
    if config.provider != "openai" and (
        config.openai_api_key or os.environ.get("OPENAI_API_KEY")
    ):
        candidates.append("openai")
    return candidates


class ProviderChain(LLMProvider):
    """Ordered list of ProviderProfiles; routes to the next on retryable
    failure."""

    def __init__(self, profiles: list[ProviderProfile]):
        self.profiles = profiles
        self._instances: dict[int, LLMProvider] = {}

    def _get_instance(self, index: int) -> LLMProvider:
        if index in self._instances:
            return self._instances[index]

        from dashscope.acli.providers import _create_provider

        profile = self.profiles[index]
        provider = _create_provider(profile)
        self._instances[index] = HardenedProvider(
            provider,
            max_retries=profile.max_retries,
        )
        return self._instances[index]

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        # Exactly ONE retry layer: HardenedProvider (see _get_instance) owns
        # per-profile retries; the chain only falls to the next profile when
        # the attempt fails with a retryable error. Non-retryable errors
        # (auth, 404, ...) stop the chain immediately.
        last_error: BaseException | None = None
        for i in range(len(self.profiles)):
            try:
                provider = self._get_instance(i)
                return await provider.chat(
                    messages,
                    tools,
                    response_format=response_format,
                )
            except Exception as e:
                last_error = e
                if not is_retryable_error(e):
                    break
                continue
        raise last_error or RuntimeError("All providers failed")

    async def chat_stream(  # pylint: disable=invalid-overridden-method
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        # Same single-retry-layer contract as chat(); HardenedProvider only
        # retries BEFORE the first chunk, but it still propagates errors that
        # occur mid-stream. So the chain may only fall to the next profile
        # when nothing was emitted yet — otherwise output would duplicate.
        last_error: BaseException | None = None
        for i in range(len(self.profiles)):
            emitted_anything = False
            try:
                provider = self._get_instance(i)
                async for chunk in provider.chat_stream(
                    messages,
                    tools,
                    response_format=response_format,
                ):
                    emitted_anything = True
                    yield chunk
                return
            except Exception as e:
                last_error = e
                if emitted_anything or not is_retryable_error(e):
                    break
                continue
        raise last_error or RuntimeError("All providers failed")
