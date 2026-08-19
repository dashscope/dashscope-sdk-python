# -*- coding: utf-8 -*-
"""Layer-1 extension system: TOML-driven custom providers + HTTP capabilities.

Users can describe new LLM providers (OpenAI-compatible) and new HTTP-based
tool capabilities in `custom-extensions.toml` without writing Python.
acli loads these at startup and folds them into PROVIDER_MODELS / the tool
registry as if they shipped with the codebase.

Two paths for finding the file (workspace overrides global on same name):
  ~/.acli/custom-extensions.toml          (global)
  ./.acli/custom-extensions.toml          (workspace)

Security model for API keys (loader enforces this — see _validate_provider):
  PREFERRED   api_key_env = "FOO_API_KEY"   # toml stores only the env var
                                            # NAME; shell provides the value.
                                            # File can safely live in git.
  FALLBACK    api_key = "ENC:..."           # XOR + machine-fingerprint
                                            # encrypted (same scheme as
                                            # config.toml's *_api_key).
                                            # Use encrypt_value to produce.
  REJECTED    api_key = "sk-plaintext"      # loader refuses, prints how to
                                            # fix.

Files written by /dev get chmod 600 so other local users can't read.
"""
# pylint: disable=too-many-branches,too-many-return-statements

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - fallback for 3.9/3.10
    import tomli as tomllib  # type: ignore

from dashscope.acli.config import CONFIG_DIR, WORKSPACE_DIR
from dashscope.acli.utils.crypto import decrypt_value
from dashscope.acli.utils.template import render_mustache_template
from dashscope.acli.utils.toml import toml_str

GLOBAL_EXTENSIONS_FILE = CONFIG_DIR / "custom-extensions.toml"
WORKSPACE_EXTENSIONS_FILE = WORKSPACE_DIR / "custom-extensions.toml"


# ===== Data classes =====


@dataclass
class CustomProvider:
    name: str
    base_url: str
    api_key_env: str = ""
    api_key_enc: str = ""  # full "ENC:..." string when present
    default_model: str = ""
    models: list[str] = field(default_factory=list)
    vision_models: list[str] = field(default_factory=list)
    audio_models: list[str] = field(default_factory=list)
    protocol: str = "openai"
    reasoning_field: str = ""
    auth: bool = True  # False for local no-auth providers (e.g. ollama)
    source: str = ""  # path of the toml it was loaded from

    def resolved_models(self) -> list[str]:
        out = [self.default_model] if self.default_model else []
        for m in self.models:
            if m and m != self.default_model:
                out.append(m)
        return out or [self.default_model or self.name]

    def resolved_protocol(self) -> str:
        """Normalize protocol aliases. 'openai_compatible' → 'openai';
        'dashscope' / 'anthropic' / 'openai' pass through unchanged."""
        p = (self.protocol or "openai").lower()
        if p == "openai_compatible":
            return "openai"
        return p

    def resolve_api_key(self, config=None) -> str:
        if not self.auth:
            return "ollama"  # placeholder; OpenAI SDK requires non-empty
        # User-set global config slot wins (set via /provider or startup).
        if config is not None:
            cfg_val = getattr(config, f"{self.name}_api_key", "")
            if cfg_val:
                return cfg_val
        if self.api_key_env:
            val = os.environ.get(self.api_key_env, "")
            if val:
                return val
        if self.api_key_enc:
            return decrypt_value(self.api_key_enc)
        return ""


@dataclass
class CustomTool:
    name: str
    description: str
    endpoint: str
    http_method: str = "POST"
    auth: str = ""  # overrides capability auth when set
    params: list[dict] = field(default_factory=list)
    body_template: str = ""  # mustache-style {{name}} placeholders
    headers: dict[str, str] = field(default_factory=dict)
    result_jsonpath: str = ""  # simple dotted path: "data.results.0.title"
    permission: str = "auto"  # auto / confirm / dangerous
    # Tool type: "http" (default, REST call) or "vision" (LLM call with
    # image content blocks). Vision tools ignore endpoint/body_template
    # and use provider+model to drive an LLM chat call.
    type: str = "http"
    provider: str = ""  # for type="vision": references [[providers]] by name
    model: str = ""  # for type="vision": vision model name to call


@dataclass
class CustomCapability:
    key: str  # e.g. "dashscope.web"
    display: str = ""
    auth: str = ""  # default auth used by tools that don't override
    api_key_env: str = ""  # optional: name of env var with the token
    api_key_enc: str = ""  # optional: "ENC:..." encrypted token, fallback
    #   when env is unset. Written via /capability enable's auto-prompt.
    tools: list[CustomTool] = field(default_factory=list)
    source: str = ""
    # In-memory reuse of a key already stored for a built-in provider whose
    # env var matches this capability's (e.g. DASHSCOPE_API_KEY ↔ tongyi).
    # Set at registration time; never persisted.
    runtime_key: str = ""

    def resolve_auth_key(self) -> str:
        """Auth token resolution: env var (named by api_key_env if set OR
        derived from auth's "bearer:$X" form) wins, then a reused built-in
        provider key, falling back to the encrypted ENC value. Empty string
        when nothing's available."""
        env_name = self.api_key_env or auth_env_name(self.auth)
        if env_name:
            val = os.environ.get(env_name, "")
            if val:
                return val
        if self.runtime_key:
            return self.runtime_key
        if self.api_key_enc:
            return decrypt_value(self.api_key_enc)
        return ""


def loaded_key_targets() -> dict | None:
    """KEY_TARGETS if acli.cli.constants is already imported, else None.

    Importing it here would run the acli.cli package init (registers every
    core tool as a side effect), so only use the already-loaded module."""
    mod = sys.modules.get("acli.cli.constants")
    return getattr(mod, "KEY_TARGETS", None) if mod is not None else None


def provider_key_for_env(
    config,
    env_name: str,
    key_targets: dict | None = None,
) -> str:
    """Key already stored for a built-in provider whose env var matches
    ``env_name`` (e.g. DASHSCOPE_API_KEY → config.tongyi_api_key). Lets an
    extension capability reuse a key the user entered once via /provider key
    instead of prompting again. Empty string when no match.

    ``key_targets`` is passed in by the caller (acli.cli.constants
    .KEY_TARGETS) — importing that module here would trigger the acli.cli
    package init and register all core tools as a side effect."""
    if not env_name or config is None or not key_targets:
        return ""
    for target in key_targets.values():
        if target.get("env") == env_name:
            return getattr(config, target["field"], "") or ""
    return ""


@dataclass
class CustomSkill:
    name: str
    description: str
    prompt_template: str
    arguments: list[str] = field(default_factory=list)
    mcp_service: str = ""
    source: str = ""


@dataclass
class CustomShellTool:
    name: str
    description: str
    command_template: str  # e.g. "curl -s {{url}}" with mustache placeholders
    params: list[dict] = field(default_factory=list)
    permission: str = "confirm"  # auto / confirm / dangerous
    source: str = ""


@dataclass
class CustomExtensions:
    providers: list[CustomProvider] = field(default_factory=list)
    capabilities: list[CustomCapability] = field(default_factory=list)
    skills: list[CustomSkill] = field(default_factory=list)
    shell_tools: list[CustomShellTool] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ===== Loading =====


def load_extensions() -> CustomExtensions:
    """Read global + workspace extension files; workspace entries with the
    same name as a global entry override (later wins). Errors are collected
    rather than raised — partial successes still register; bad entries get
    surfaced to the user via the cli banner."""
    ext = CustomExtensions()
    by_provider: dict[str, CustomProvider] = {}
    by_capability: dict[str, CustomCapability] = {}
    by_skill: dict[str, CustomSkill] = {}
    by_shell_tool: dict[str, CustomShellTool] = {}

    # Load order (later wins on same name): global → workspace.
    # Bundled defaults were removed; users declare providers in
    # ~/.acli/custom-extensions.toml or ./.acli/custom-extensions.toml.
    # See the basic-chat example (`acli example download basic-chat`).
    paths: list[Path] = [GLOBAL_EXTENSIONS_FILE, WORKSPACE_EXTENSIONS_FILE]

    for path in paths:
        if not path.exists():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            ext.errors.append(f"{path}: TOML parse error: {e}")
            continue
        except OSError as e:
            ext.errors.append(f"{path}: read failed: {e}")
            continue
        for raw in data.get("providers", []):
            err = _validate_provider(raw)
            if err:
                ext.errors.append(
                    f"{path}: provider {raw.get('name', '?')}: {err}",
                )
                continue
            p = CustomProvider(
                name=raw["name"],
                base_url=raw["base_url"],
                api_key_env=raw.get("api_key_env", "") or "",
                api_key_enc=(
                    raw.get("api_key", "")
                    if str(raw.get("api_key", "")).startswith("ENC:")
                    else ""
                ),
                default_model=raw.get("default_model", "") or "",
                models=list(raw.get("models", []) or []),
                vision_models=list(raw.get("vision_models", []) or []),
                audio_models=list(raw.get("audio_models", []) or []),
                protocol=raw.get("protocol", "openai"),
                reasoning_field=raw.get("reasoning_field", "") or "",
                auth=bool(raw.get("auth", True)),
                source=str(path),
            )
            by_provider[p.name] = p

        for raw in data.get("capabilities", []):
            err = _validate_capability(raw)
            if err:
                ext.errors.append(
                    f"{path}: capability {raw.get('key', '?')}: {err}",
                )
                continue
            cap = CustomCapability(
                key=raw["key"],
                display=raw.get("display", raw["key"]),
                auth=raw.get("auth", ""),
                api_key_env=raw.get("api_key_env", "") or "",
                api_key_enc=(
                    raw.get("api_key", "")
                    if str(raw.get("api_key", "")).startswith("ENC:")
                    else ""
                ),
                source=str(path),
            )
            for traw in raw.get("tools", []) or []:
                terr = _validate_tool(traw)
                if terr:
                    ext.errors.append(
                        f"{path}: capability {cap.key} tool "
                        f"{traw.get('name', '?')}: {terr}",
                    )
                    continue
                cap.tools.append(
                    CustomTool(
                        name=traw["name"],
                        description=traw.get("description", ""),
                        endpoint=traw.get("endpoint", ""),
                        http_method=traw.get("http_method", "POST").upper(),
                        auth=traw.get("auth", ""),
                        params=list(traw.get("params", []) or []),
                        body_template=traw.get("body_template", ""),
                        headers=dict(traw.get("headers", {}) or {}),
                        result_jsonpath=traw.get("result_jsonpath", ""),
                        permission=traw.get("permission", "auto").lower(),
                        type=(traw.get("type", "http") or "http").lower(),
                        provider=traw.get("provider", "") or "",
                        model=traw.get("model", "") or "",
                    ),
                )
            by_capability[cap.key] = cap

        for raw in data.get("skills", []):
            if not raw.get("name") or not raw.get("prompt_template"):
                ext.errors.append(
                    f"{path}: skill {raw.get('name', '?')}: "
                    f"missing name or prompt_template",
                )
                continue
            by_skill[raw["name"]] = CustomSkill(
                name=raw["name"],
                description=raw.get("description", ""),
                prompt_template=raw["prompt_template"],
                arguments=list(raw.get("arguments", []) or []),
                mcp_service=raw.get("mcp_service", "") or "",
                source=str(path),
            )

        for raw in data.get("shell_tools", []):
            if not raw.get("name") or not raw.get("command_template"):
                ext.errors.append(
                    f"{path}: shell_tool {raw.get('name', '?')}: "
                    f"missing name or command_template",
                )
                continue
            by_shell_tool[raw["name"]] = CustomShellTool(
                name=raw["name"],
                description=raw.get("description", ""),
                command_template=raw["command_template"],
                params=list(raw.get("params", []) or []),
                permission=raw.get("permission", "confirm").lower(),
                source=str(path),
            )

    ext.providers = list(by_provider.values())
    ext.capabilities = list(by_capability.values())
    ext.skills = list(by_skill.values())
    ext.shell_tools = list(by_shell_tool.values())
    return ext


def _validate_provider(raw: dict) -> str:
    if not raw.get("name"):
        return "missing 'name'"
    if not raw.get("base_url"):
        return "missing 'base_url'"
    protocol = str(raw.get("protocol", "openai")).lower()
    if protocol not in (
        "openai",
        "openai_compatible",
        "anthropic",
        "dashscope",
    ):
        return (
            "protocol must be 'openai', 'openai_compatible', "
            "'anthropic', or 'dashscope'"
        )
    auth = raw.get("auth", True)
    if not auth:
        # No-auth providers (e.g. ollama) skip key validation entirely.
        return ""
    raw_key = raw.get("api_key", "")
    has_env = bool(raw.get("api_key_env"))
    has_enc = isinstance(raw_key, str) and raw_key.startswith("ENC:")
    if raw_key and not has_enc:
        return (
            "plaintext api_key in toml is refused for security. "
            'Use api_key_env = "FOO_API_KEY" (preferred) or encrypt the '
            'value first and store as api_key = "ENC:...".'
        )
    if not (has_env or has_enc):
        return (
            'no API key source: set api_key_env = "FOO_API_KEY" '
            'or store an encrypted api_key = "ENC:..." '
            "(or auth = false for no-auth providers)."
        )
    return ""


def _validate_capability(raw: dict) -> str:
    key = raw.get("key", "")
    if not key or "." not in key:
        return "key must be in 'vendor.feature' form (e.g. dashscope.web)"
    raw_key = raw.get("api_key", "")
    if raw_key and not (
        isinstance(raw_key, str) and raw_key.startswith("ENC:")
    ):
        return (
            "plaintext api_key in toml is refused for security. "
            'Use api_key_env = "FOO_TOKEN" or run /capability enable '
            "to set the key."
        )
    return ""


def auth_env_name(auth: str) -> str:
    """Extract the env-var name an auth spec references, or empty if the
    spec is 'none' / unset / malformed.

      bearer:$FOO_TOKEN              -> FOO_TOKEN
      apikey-header:X-API-KEY:$FOO   -> FOO
    """
    if not auth or auth == "none":
        return ""
    if auth.startswith("bearer:$"):
        return auth[len("bearer:$") :]
    if auth.startswith("apikey-header:"):
        try:
            _, _, env_part = auth.split(":", 2)
        except ValueError:
            return ""
        return env_part.lstrip("$")
    return ""


def _validate_tool(raw: dict) -> str:
    if not raw.get("name"):
        return "missing 'name'"
    tool_type = (raw.get("type", "http") or "http").lower()
    if tool_type == "vision":
        if not raw.get("provider"):
            return "type='vision' requires 'provider' (a [[providers]] name)"
        if not raw.get("model"):
            return "type='vision' requires 'model' (vision model name)"
        return ""
    # Default: HTTP REST tool
    if not raw.get("endpoint"):
        return "missing 'endpoint'"
    if raw.get("http_method", "POST").upper() not in (
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
    ):
        return "http_method must be GET/POST/PUT/DELETE/PATCH"
    auth = raw.get("auth", "") or ""
    if auth and not (
        auth.startswith("bearer:$")
        or auth.startswith("apikey-header:")
        or auth == "none"
    ):
        return (
            "auth must be 'none', 'bearer:$ENV', or 'apikey-header:NAME:$ENV'"
        )
    return ""


# ===== Provider registration =====


def merge_providers_into_catalog(
    ext: CustomExtensions,
    provider_models: dict[str, list[str]],
) -> dict[str, CustomProvider]:
    """Fold custom providers into PROVIDER_MODELS. Returns a name → spec
    map so get_provider can later look up base_url + key resolver."""
    out: dict[str, CustomProvider] = {}
    for p in ext.providers:
        provider_models[p.name] = p.resolved_models()
        out[p.name] = p
    return out


# ===== HTTP tool factory =====


def _extract_jsonpath(data: Any, path: str) -> Any:
    """Tiny dotted-path extractor: "data.results.0.title" walks down a mix
    of dicts and lists. Returns the original data unchanged when path is
    empty or any segment fails to resolve (caller can see the full body).
    Intentionally simple — no wildcards / filters / recursion. If users
    need real JSONPath we can swap in jsonpath-ng later."""
    if not path:
        return data
    cur = data
    for seg in path.split("."):
        if isinstance(cur, dict):
            if seg in cur:
                cur = cur[seg]
            else:
                return data  # path failed, return full body
        elif isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                return data
        else:
            return data
    return cur


def _render_url_template(template: str, params: dict[str, Any]) -> str:
    """{{var}} substitution for URLs: values are percent-encoded, never
    JSON-quoted (render_mustache_template would inject literal quotes)."""
    from urllib.parse import quote

    out = template
    for k, v in params.items():
        encoded = quote(str(v), safe="")
        out = out.replace("{{" + k + "}}", encoded)
        out = out.replace("{{ " + k + " }}", encoded)
    return out


def _resolve_auth_header(
    auth: str,
    fallback_key: str = "",
) -> tuple[str, str] | None:
    """Translate a capability/tool auth spec into a (header_name, value)
    pair. `fallback_key` is used when the env var named by the auth spec
    is unset (typical source: capability.resolve_auth_key() returning the
    decrypted api_key_enc). Returns None for 'none' / empty / unresolvable
    (caller decides fail vs unauth)."""
    if not auth or auth == "none":
        return None
    if auth.startswith("bearer:$"):
        env = auth[len("bearer:$") :]
        val = os.environ.get(env, "") or fallback_key
        return ("Authorization", f"Bearer {val}") if val else None
    if auth.startswith("apikey-header:"):
        try:
            _, header, env_part = auth.split(":", 2)
        except ValueError:
            return None
        env = env_part.lstrip("$")
        val = os.environ.get(env, "") or fallback_key
        return (header, val) if val else None
    return None


def build_http_tool(cap: CustomCapability, tool: CustomTool):
    """Return an async callable that the registry can invoke. Signature is
    `async def _call(**kwargs) -> str` matching the rest of acli's tools."""
    import httpx

    async def _call(**kwargs) -> str:
        # Apply declared param defaults so omitted optional args don't leave
        # unrendered {{placeholders}} that break the body JSON.
        for p in tool.params:
            pname = p.get("name")
            if pname and pname not in kwargs and "default" in p:
                kwargs[pname] = p["default"]
        try:
            body_str = (
                render_mustache_template(tool.body_template, kwargs)
                if tool.body_template
                else ""
            )
            body_dict = json.loads(body_str) if body_str.strip() else None
            endpoint = _render_url_template(tool.endpoint, kwargs)
        except json.JSONDecodeError as e:
            return (
                f"Error: body template did not render to valid JSON "
                f"({e}); template snippet: {body_str[:200]!r}"
            )

        headers = dict(tool.headers)
        auth_spec = tool.auth or cap.auth
        # cap-level fallback: env miss → decrypted api_key_enc → empty
        fallback = cap.resolve_auth_key()
        pair = _resolve_auth_header(auth_spec, fallback_key=fallback)
        if pair is None and not auth_spec:
            # Capability declares credentials (api_key_env / api_key_enc) but
            # no auth spec — default to Bearer so the token actually reaches
            # the API instead of sending an unauthenticated request (401).
            if fallback:
                pair = ("Authorization", f"Bearer {fallback}")
            elif cap.api_key_env or cap.api_key_enc:
                return (
                    f"Error: tool {tool.name} requires credentials "
                    f"(env: {cap.api_key_env or 'not configured'}), "
                    f"but the env var is unset and no encrypted key "
                    f"exists in toml. Run /capability enable {cap.key} "
                    f"to enter one."
                )
        if pair:
            headers[pair[0]] = pair[1]
        elif auth_spec and auth_spec != "none":
            return (
                f"Error: tool {tool.name} requires credentials "
                f"({auth_spec}), but the env var is unset and no "
                f"encrypted key exists in toml. Run "
                f"[bold]/capability enable {cap.key}[/bold] to enter one."
            )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                send_body = (
                    body_dict
                    if tool.http_method.upper() not in ("GET", "DELETE")
                    else None
                )
                resp = await client.request(
                    tool.http_method,
                    endpoint,
                    json=send_body,
                    headers=headers,
                )
        except Exception as e:
            return f"Error: HTTP call failed: {type(e).__name__}: {e}"

        if resp.status_code >= 400:
            return f"Error: HTTP {resp.status_code}: {resp.text[:500]}"

        try:
            data = resp.json()
        except ValueError:
            return resp.text[:5000]

        data = _extract_jsonpath(data, tool.result_jsonpath)
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    return _call


def build_vision_tool(cap: CustomCapability, tool: CustomTool):
    """Return an async callable that drives an LLM vision call.

    Reads the image file → base64 data URL → builds a multimodal user
    message → calls the referenced provider's chat() → returns the text
    response. The provider lookup reuses the existing [[providers]] block
    for base_url/protocol/api_key_env, so vision capabilities don't need
    to redeclare connection settings.
    """
    provider_name = tool.provider
    model_name = tool.model

    async def _call(**kwargs) -> str:
        from dashscope.acli.providers import _create_provider
        from dashscope.acli.providers.profile import ProviderProfile
        from dashscope.acli.utils.images import image_to_data_url
        from dashscope.acli.utils.paths import validate_path

        image_path = kwargs.get("image_path", "")
        question = kwargs.get("question") or "Describe this image"

        if not image_path:
            return "Error: missing image_path parameter"
        try:
            safe_path = validate_path(image_path)
            data_url = image_to_data_url(safe_path)
        except ValueError as e:
            return f"Error: failed to read image: {e}"
        except OSError as e:
            return f"Error: failed to read image: {e}"

        ext = find_provider(provider_name)
        api_key = ext.resolve_api_key() if ext else ""
        # Fall back to the capability's own auth if the provider has no key
        # (lets a vision capability carry its own api_key_env).
        if not api_key:
            api_key = cap.resolve_auth_key()
        base_url = ext.base_url if ext else None
        protocol = ext.resolved_protocol() if ext else "openai"
        profile = ProviderProfile(
            name=provider_name,
            provider=provider_name,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
        )
        try:
            provider = _create_provider(profile)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ]
            response = await provider.chat(messages)
            return response.content
        except Exception as e:
            return f"Vision model call failed: {e}"

    return _call


def tool_parameters_schema(tool: CustomTool) -> dict:
    """Build the JSON schema acli's registry expects from a CustomTool's
    params list. Maps {name, type, required, description, default} entries
    to standard JSON-Schema properties + required[]."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in tool.params:
        name = p.get("name")
        if not name:
            continue
        prop: dict[str, Any] = {"type": p.get("type", "string")}
        if p.get("description"):
            prop["description"] = p["description"]
        if "default" in p:
            prop["default"] = p["default"]
        properties[name] = prop
        if p.get("required"):
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ===== TOML writing (for /dev add) =====


def append_provider(target: Path, p: CustomProvider) -> None:
    """Append a [[providers]] block to target toml; create file with 0600
    perms if it doesn't exist."""
    lines = []
    if target.exists():
        existing = target.read_text(encoding="utf-8").rstrip()
        if existing:
            lines.append(existing)
            lines.append("")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines.append("# acli custom extensions — providers and capabilities")
        lines.append("# Loader: src/acli/extensions.py")
        lines.append("")

    lines.append("[[providers]]")
    lines.append(f'name = "{p.name}"')
    lines.append(f'base_url = "{p.base_url}"')
    if p.api_key_env:
        lines.append(f'api_key_env = "{p.api_key_env}"')
    elif p.api_key_enc:
        lines.append(f'api_key = "{p.api_key_enc}"')
    if p.default_model:
        lines.append(f'default_model = "{p.default_model}"')
    if p.models:
        models_lit = ", ".join(f'"{m}"' for m in p.models)
        lines.append(f"models = [{models_lit}]")
    if p.protocol and p.protocol != "openai":
        lines.append(f'protocol = "{p.protocol}"')
    if not p.auth:
        lines.append("auth = false")
    if p.reasoning_field:
        lines.append(f'reasoning_field = "{p.reasoning_field}"')

    _write_private_text(target, "\n".join(lines) + "\n")


def remove_provider(target: Path, name: str) -> bool:
    """Remove the [[providers]] block whose name matches. Returns True if
    something was removed."""
    if not target.exists():
        return False
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False
    providers = data.get("providers", [])
    new_providers = [p for p in providers if p.get("name") != name]
    if len(new_providers) == len(providers):
        return False
    data["providers"] = new_providers
    _write_full(target, data)
    return True


def append_capability_scaffold(
    target: Path,
    key: str,
    display: str = "",
) -> None:
    """Write a commented [[capabilities]] template the user can fill in
    with their text editor. Includes one example tool block."""
    lines = []
    if target.exists():
        existing = target.read_text(encoding="utf-8").rstrip()
        if existing:
            lines.append(existing)
            lines.append("")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines.append("# acli custom extensions — providers and capabilities")
        lines.append("")

    lines.append("[[capabilities]]")
    lines.append(
        f'key = "{key}"                       # required, vendor.feature form',
    )
    lines.append(f'display = "{display or key}"')
    lines.append(
        "# default auth used by tools that omit their own; "
        "comment out for none",
    )
    lines.append('# auth = "bearer:$YOUR_API_KEY_ENV"')
    lines.append("")
    lines.append("[[capabilities.tools]]")
    lines.append(
        'name = "example_tool"               # function name LLM will see',
    )
    lines.append(
        'description = "describe what this tool does and when to call it"',
    )
    lines.append('endpoint = "https://api.example.com/v1/something"')
    lines.append(
        'http_method = "POST"                # GET/POST/PUT/DELETE/PATCH',
    )
    lines.append(
        'permission = "auto"                 # auto / confirm / dangerous',
    )
    lines.append("# Optional per-tool auth (overrides capability default):")
    lines.append('# auth = "bearer:$OTHER_ENV"')
    lines.append("params = [")
    lines.append(
        '  {name="query", type="string", required=true, '
        'description="search query"},',
    )
    lines.append(
        '  {name="count", type="integer", default=10, '
        'description="result count"},',
    )
    lines.append("]")
    lines.append(
        "# body_template: mustache-style {{var}} substitution; "
        "values JSON-encoded.",
    )
    lines.append(
        'body_template = \'{"query": {{query}}, "count": {{count}}}\'',
    )
    lines.append("# Optional: extract a sub-field of the JSON response")
    lines.append('# result_jsonpath = "data.results"')
    lines.append("")

    _write_private_text(target, "\n".join(lines) + "\n")


def set_capability_secret(
    target: Path,
    cap_key: str,
    api_key_env: str = "",
    api_key_enc: str = "",
) -> bool:
    """Set / overwrite the api_key_env or api_key (ENC) field on an existing
    [[capabilities]] block. Returns True on success, False if cap_key not
    found. Pass empty strings to clear fields (e.g. switch from env to ENC).
    """
    if not target.exists():
        return False
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False
    caps = data.get("capabilities", [])
    found = False
    for cap in caps:
        if cap.get("key") != cap_key:
            continue
        found = True
        if api_key_env:
            cap["api_key_env"] = api_key_env
            cap.pop("api_key", None)
        elif api_key_enc:
            cap["api_key"] = api_key_enc
            cap.pop("api_key_env", None)
        else:
            cap.pop("api_key_env", None)
            cap.pop("api_key", None)
        break
    if not found:
        return False
    data["capabilities"] = caps
    _write_full(target, data)
    return True


def set_provider_secret(
    target: Path,
    provider_name: str,
    api_key_env: str = "",
    api_key_enc: str = "",
) -> bool:
    """Set / overwrite the api_key_env or api_key (ENC) field on an existing
    [[providers]] block. Returns True on success, False if provider not found.

    Writing api_key_enc keeps any existing api_key_env so the env var still
    wins when set (resolve_api_key falls back to ENC only when env is empty).
    Pass empty strings to clear both fields.
    """
    if not target.exists():
        return False
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False
    providers = data.get("providers", [])
    found = False
    for p in providers:
        if p.get("name") != provider_name:
            continue
        found = True
        if api_key_env:
            p["api_key_env"] = api_key_env
            p.pop("api_key", None)
        elif api_key_enc:
            p["api_key"] = api_key_enc
            # keep api_key_env so env var still wins when set
        else:
            p.pop("api_key_env", None)
            p.pop("api_key", None)
        break
    if not found:
        return False
    data["providers"] = providers
    _write_full(target, data)
    return True


def remove_capability(target: Path, key: str) -> bool:
    if not target.exists():
        return False
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False
    caps = data.get("capabilities", [])
    new_caps = [c for c in caps if c.get("key") != key]
    if len(new_caps) == len(caps):
        return False
    data["capabilities"] = new_caps
    _write_full(target, data)
    return True


def append_skill(target: Path, s: CustomSkill) -> None:
    """Append a [[skills]] block to target toml."""
    lines = []
    if target.exists():
        existing = target.read_text(encoding="utf-8").rstrip()
        if existing:
            lines.append(existing)
            lines.append("")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines.append("# acli custom extensions")
        lines.append("")

    lines.append("[[skills]]")
    lines.append(f"name = {toml_str(s.name)}")
    lines.append(f"description = {toml_str(s.description)}")
    lines.append(f"prompt_template = {toml_str(s.prompt_template)}")
    if s.arguments:
        args_lit = ", ".join(toml_str(a) for a in s.arguments)
        lines.append(f"arguments = [{args_lit}]")
    if s.mcp_service:
        lines.append(f"mcp_service = {toml_str(s.mcp_service)}")

    _write_private_text(target, "\n".join(lines) + "\n")


def remove_skill(target: Path, name: str) -> bool:
    if not target.exists():
        return False
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False
    skills = data.get("skills", [])
    new_skills = [s for s in skills if s.get("name") != name]
    if len(new_skills) == len(skills):
        return False
    data["skills"] = new_skills
    _write_full(target, data)
    return True


def append_shell_tool(target: Path, t: CustomShellTool) -> None:
    """Append a [[shell_tools]] block to target toml."""
    lines = []
    if target.exists():
        existing = target.read_text(encoding="utf-8").rstrip()
        if existing:
            lines.append(existing)
            lines.append("")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines.append("# acli custom extensions")
        lines.append("")

    lines.append("[[shell_tools]]")
    lines.append(f"name = {toml_str(t.name)}")
    lines.append(f"description = {toml_str(t.description)}")
    lines.append(f"command_template = {toml_str(t.command_template)}")
    if t.params:
        params_lit = ", ".join(
            "{"
            + ", ".join(
                (
                    f"{k}={toml_str(v)}"
                    if isinstance(v, str)
                    else f"{k}={str(v).lower() if isinstance(v, bool) else v}"
                )
                for k, v in p.items()
            )
            + "}"
            for p in t.params
        )
        lines.append(f"params = [{params_lit}]")
    if t.permission != "confirm":
        lines.append(f"permission = {toml_str(t.permission)}")

    _write_private_text(target, "\n".join(lines) + "\n")


def remove_shell_tool(target: Path, name: str) -> bool:
    if not target.exists():
        return False
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False
    tools = data.get("shell_tools", [])
    new_tools = [t for t in tools if t.get("name") != name]
    if len(new_tools) == len(tools):
        return False
    data["shell_tools"] = new_tools
    _write_full(target, data)
    return True


def _write_full(target: Path, data: dict) -> None:
    """Minimal TOML emitter for our schema. We don't try to be a
    general-purpose serializer — we only round-trip what load_extensions
    produces."""
    lines = ["# acli custom extensions", ""]
    for p in data.get("providers", []):
        lines.append("[[providers]]")
        for k, v in p.items():
            lines.append(_toml_kv(k, v))
        lines.append("")
    for c in data.get("capabilities", []):
        lines.append("[[capabilities]]")
        for k, v in c.items():
            if k == "tools":
                continue
            lines.append(_toml_kv(k, v))
        for t in c.get("tools", []):
            lines.append("[[capabilities.tools]]")
            for k, v in t.items():
                lines.append(_toml_kv(k, v))
        lines.append("")
    for s in data.get("skills", []):
        lines.append("[[skills]]")
        for k, v in s.items():
            lines.append(_toml_kv(k, v))
        lines.append("")
    for t in data.get("shell_tools", []):
        lines.append("[[shell_tools]]")
        for k, v in t.items():
            lines.append(_toml_kv(k, v))
        lines.append("")
    _write_private_text(target, "\n".join(lines).rstrip() + "\n")


def _toml_kv(k: str, v: Any) -> str:
    if isinstance(v, bool):
        return f"{k} = {'true' if v else 'false'}"
    if isinstance(v, (int, float)):
        return f"{k} = {v}"
    if isinstance(v, list):
        # Naive: assume list of primitives or list of inline tables (params).
        if v and isinstance(v[0], dict):
            inline = []
            for item in v:
                pairs = []
                for ik, iv in item.items():
                    if isinstance(iv, str):
                        pairs.append(f"{ik}={toml_str(iv)}")
                    elif isinstance(iv, bool):
                        pairs.append(f"{ik}={'true' if iv else 'false'}")
                    else:
                        pairs.append(f"{ik}={iv}")
                inline.append("{" + ", ".join(pairs) + "}")
            return f"{k} = [{', '.join(inline)}]"
        rendered = ", ".join(
            toml_str(x) if isinstance(x, str) else str(x) for x in v
        )
        return f"{k} = [{rendered}]"
    if isinstance(v, dict):
        # Best-effort inline table
        pairs = ", ".join(f"{ik}={toml_str(iv)}" for ik, iv in v.items())
        return f"{k} = {{ {pairs} }}"
    return f"{k} = {toml_str(v)}"


def _chmod_user_only(path: Path) -> None:
    """Make sure the file is 0600 — other local users shouldn't be able
    to read API-key-bearing config."""
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # not fatal; e.g. NFS or unusual filesystems


def _write_private_text(path: Path, text: str) -> None:
    """Write text with 0600 perms from the moment the file exists.

    write_text()+chmod leaves a window where a newly created file holding
    an ENC: API key is readable under the default umask; passing the mode
    to os.open closes it.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    _chmod_user_only(path)


# ===== Encryption helper for users who want api_key in toml =====


def encrypt_for_toml(plaintext: str) -> str:
    """Wrap utils.crypto.encrypt_value for /dev's interactive flow when the
    user chooses ENC fallback instead of env. Returns the full 'ENC:...'
    string ready to drop into api_key field."""
    from dashscope.acli.utils.crypto import encrypt_value

    return encrypt_value(plaintext)


# ===== Process-wide singleton + apply() hook =====
#
# get_provider() and tools/platform.py need to look up extension specs at
# runtime. Stash the most recent load() result here so they can consult it
# without each one re-reading the TOML.

_current: CustomExtensions | None = None


def current() -> CustomExtensions:
    """Return the currently-loaded extensions (or an empty stub if apply()
    has never run)."""
    return _current if _current is not None else CustomExtensions()


def find_provider(name: str) -> CustomProvider | None:
    for p in current().providers:
        if p.name == name:
            return p
    return None


def find_capability(key: str) -> CustomCapability | None:
    for c in current().capabilities:
        if c.key == key:
            return c
    return None


def apply_extensions(
    provider_models: dict[str, list[str]],
) -> CustomExtensions:
    """Load extensions and fold them into PROVIDER_MODELS. Capability
    registration is handled inside tools/platform.py by consulting
    current() — kept there so it ties into the same tracking that the
    existing platform capabilities use."""
    global _current
    ext = load_extensions()
    merge_providers_into_catalog(ext, provider_models)
    _register_custom_skills(ext)
    _register_custom_shell_tools(ext)
    from dashscope.acli.skills.base import load_skill_files

    load_skill_files()
    _current = ext
    _sync_modality_sets(ext)
    return ext


def _sync_modality_sets(ext: CustomExtensions) -> None:
    """Populate acli.config.VISION_MODELS / AUDIO_MODELS so is_vision_model()
    / is_audio_model() work for the @image.png and @audio.mp3 REPL flows.

    Vision models come from two sources now:
      1. [[capabilities.tools]] with type="vision" — the tool's `model` field
         is added to VISION_MODELS (so the main agent knows its model
         supports image input when that model is the active main model).
      2. [[providers]] vision_models lists — retained for backwards
         compatibility with tomls that haven't migrated to capabilities.
    """
    from dashscope.acli.config import AUDIO_MODELS, VISION_MODELS

    vision = set()
    audio = set()
    for prov in ext.providers:
        vision.update(prov.vision_models)
        audio.update(prov.audio_models)
    for cap in ext.capabilities:
        for tool in cap.tools:
            if tool.type == "vision" and tool.model:
                vision.add(tool.model)
    VISION_MODELS.clear()
    VISION_MODELS.update(vision)
    AUDIO_MODELS.clear()
    AUDIO_MODELS.update(audio)


def _register_custom_skills(ext: CustomExtensions) -> None:
    """Register custom skills from extensions into the skill registry."""
    from dashscope.acli.skills.base import Skill, register

    for s in ext.skills:
        register(
            Skill(
                name=s.name,
                description=s.description,
                mcp_service=s.mcp_service,
                prompt_template=s.prompt_template,
                arguments=s.arguments,
            ),
        )


def _register_custom_shell_tools(ext: CustomExtensions) -> None:
    """Register custom shell-command tools from extensions into the tool
    registry."""
    import subprocess as sp

    from dashscope.acli.tools.registry import (
        PermissionLevel,
        ToolDefinition,
        registry,
    )

    for t in ext.shell_tools:
        perm = getattr(
            PermissionLevel,
            t.permission.upper(),
            PermissionLevel.CONFIRM,
        )
        params_schema: dict = {"type": "object", "properties": {}}
        required: list[str] = []
        for p in t.params:
            name = p.get("name", "")
            if not name:
                continue
            params_schema["properties"][name] = {
                "type": p.get("type", "string"),
                "description": p.get("description", ""),
            }
            if p.get("required"):
                required.append(name)
        if required:
            params_schema["required"] = required

        def _make_fn(cmd_template: str):
            async def _run(**kwargs) -> str:
                import shlex

                cmd = cmd_template
                for k, v in kwargs.items():
                    # shell-quote every substitution: tool-call arguments come
                    # from the model, raw interpolation is command injection.
                    cmd = cmd.replace("{{" + k + "}}", shlex.quote(str(v)))
                try:
                    result = sp.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    output = result.stdout
                    if result.returncode != 0 and result.stderr:
                        output += f"\n[stderr] {result.stderr}"
                    return output.strip() or "(no output)"
                except sp.TimeoutExpired:
                    return "Command timed out (60s)"
                except Exception as e:
                    return f"Execution failed: {e}"

            return _run

        registry.register(
            ToolDefinition(
                name=t.name,
                description=t.description,
                permission=perm,
                func=_make_fn(t.command_template),
                parameters=params_schema,
            ),
        )
