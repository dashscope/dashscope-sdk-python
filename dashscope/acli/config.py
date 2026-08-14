# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dashscope.acli.utils.crypto import decrypt_value, encrypt_value
from dashscope.acli.utils.paths import atomic_write_text
from dashscope.acli.utils.toml import (
    load_toml,
    parse_toml_inline_table,
    toml_str,
)

CONFIG_DIR = Path.home() / ".acli"
CONFIG_FILE = CONFIG_DIR / "config.toml"

WORKSPACE_DIR = Path.cwd() / ".acli"
WORKSPACE_CONFIG_FILE = WORKSPACE_DIR / "config.toml"
WORKSPACE_SYSTEM_PROMPT_FILE = WORKSPACE_DIR / "system-prompt.md"

PROVIDER_MODELS = {
    "tongyi": [
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.5-plus",
        "qwen3.5-flash",
        "qwen3-max",
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "qwen-vl-max",
        "qwen-vl-plus",
        "qwen-omni-turbo",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3-mini"],
}

PROVIDERS = list(PROVIDER_MODELS.keys())

# Context-window sizes (tokens) by model-name prefix, longest prefix wins so
# e.g. "qwen-turbo" beats "qwen". Unknown models fall back to DEFAULT below.
MODEL_CONTEXT_WINDOWS = {
    "qwen-turbo": 1_000_000,
    "qwen": 128_000,
    "claude": 200_000,
    "gpt-4o": 128_000,
    "o1": 200_000,
    "o3": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000


def context_window_for_model(model: str) -> int:
    """Return the context window (tokens) for a model name.

    Prefix-matched against MODEL_CONTEXT_WINDOWS (longest prefix first) so
    versioned aliases (qwen-turbo-2025-04-28, gpt-4o-2024-...) resolve; unknown
    models get the conservative default.
    """
    for prefix in sorted(MODEL_CONTEXT_WINDOWS, key=len, reverse=True):
        if model.startswith(prefix):
            return MODEL_CONTEXT_WINDOWS[prefix]
    return DEFAULT_CONTEXT_WINDOW


def normalize_model_name(model: str) -> str:
    """API model IDs are case-sensitive and conventionally lowercase;
    normalize before register/switch."""
    return model.strip().lower()


def register_custom_model(config: "Config", provider: str, model: str) -> str:
    """Register a custom model: normalize, merge into PROVIDER_MODELS,
    persist to workspace."""
    model = normalize_model_name(model)
    if provider in PROVIDER_MODELS and model not in PROVIDER_MODELS[provider]:
        PROVIDER_MODELS[provider].append(model)
    entry = f"{provider}:{model}"
    custom = getattr(config, "custom_models", None)
    if custom is not None and entry not in custom:
        custom.append(entry)
        config.save_workspace()
    return model


# Models that accept image content blocks in chat messages. Used to gate
# @image.png input — non-vision models would get an API error otherwise.
# Prefix-matched so versioned aliases (qwen-vl-max-latest,
# gpt-4o-2024-...) work.
# Populated at runtime by apply_extensions() from the vision_models lists
# declared on each [[providers]] block in custom-extensions.toml.
VISION_MODELS: set[str] = set()

# Models that accept audio content blocks (OpenAI input_audio format).
AUDIO_MODELS: set[str] = set()


def _is_model_in_set(model: str, prefixes: set[str]) -> bool:
    """True if the model name starts with any registered prefix."""
    return any(model.startswith(p) for p in prefixes)


def is_vision_model(model: str) -> bool:
    """True if the model name starts with any registered vision-model
    prefix."""
    return _is_model_in_set(model, VISION_MODELS)


def is_audio_model(model: str) -> bool:
    """True if the model name starts with any registered audio-model prefix."""
    return _is_model_in_set(model, AUDIO_MODELS)


def _valid_models_for_provider(
    provider: str,
    custom_models: list[str],
) -> list[str]:
    """Return ordered valid model names for a provider.

    Includes built-in models, extension provider models from
    custom-extensions.toml, and custom_models entries of the form
    ``provider:model``.
    """
    seen: set[str] = set()
    valid: list[str] = []

    def _add(m: str) -> None:
        if m and m not in seen:
            seen.add(m)
            valid.append(m)

    for m in PROVIDER_MODELS.get(provider, []):
        _add(m)

    try:
        from dashscope.acli.extensions import load_extensions

        ext = load_extensions()
        for p in ext.providers:
            if p.name == provider:
                for m in p.resolved_models():
                    _add(m)
                break
    except Exception:
        pass

    for entry in custom_models:
        if ":" in entry:
            prov, m = entry.split(":", 1)
            if prov.strip() == provider:
                _add(m.strip())

    return valid


@dataclass
class MCPServerConfig:
    service: str
    url: str = ""


@dataclass
class SubagentConfig:
    """Per-subagent configuration overrides (model, temperature, max_turns)."""

    model: str = ""
    temperature: float = 0.0
    max_turns: int = 0

    def to_dict(self) -> dict:
        d = {}
        if self.model:
            d["model"] = self.model
        if self.temperature:
            d["temperature"] = self.temperature
        if self.max_turns:
            d["max_turns"] = self.max_turns
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SubagentConfig:
        return cls(
            model=d.get("model", ""),
            temperature=float(d.get("temperature", 0.0)),
            max_turns=int(d.get("max_turns", 0)),
        )


@dataclass
class DelegationConfig:
    """Configuration for the delegate tool."""

    max_concurrent: int = 5
    default_timeout: int = 120
    allow_nested: bool = False

    def to_dict(self) -> dict:
        return {
            "max_concurrent": self.max_concurrent,
            "default_timeout": self.default_timeout,
            "allow_nested": self.allow_nested,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DelegationConfig:
        return cls(
            max_concurrent=int(d.get("max_concurrent", 5)),
            default_timeout=int(d.get("default_timeout", 120)),
            allow_nested=str(d.get("allow_nested", False)).lower()
            in ("true", "1", "yes"),
        )


@dataclass
class Config:
    provider: str = "tongyi"
    model: str = "qwen3.7-plus"
    # Dual-LLM: when thinking_model is set, Plan/Thinking loop phases route to
    # the thinking model; Execute uses the execution model (provider/model
    # above).
    # Empty = single-model mode (default, backward compatible).
    thinking_provider: str = ""
    thinking_model: str = ""
    # Loop mode: "auto" = LLM-driven flat loop (default); "structured" =
    # 6-phase closed cycle (Target→Plan→Execute→Evaluate→Feedback→Thinking).
    loop_mode: str = "auto"
    tongyi_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    base_url: str = ""
    auto_approve: bool = False
    max_turns: int = 50
    timeout: int = 30
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    memory_enabled: bool = True
    memory_user_id: str = ""
    memory_library_id: str = ""
    platform: str = "bailian"
    user_name: str = ""
    enabled_capabilities: list[str] | None = None
    custom_models: list[str] = field(default_factory=list)
    user_directives: list[str] = field(default_factory=list)
    session_persist: bool = True
    asr_model: str = "paraformer-realtime-v2"
    voice_silence_duration: float = 2.0
    voice_max_seconds: int = 60
    voice_silence_threshold: int = 500
    tts_enabled: bool = False
    tts_model: str = "cosyvoice-v2"
    tts_voice: str = "longxiaochun_v2"
    tts_speed: float = 1.0
    subagents: dict[str, SubagentConfig] = field(default_factory=dict)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    theme: dict[str, str] = field(
        default_factory=lambda: {
            "background": "#ffffff",
            "text": "#1e1e1e",
            "border": "ansi_blue",
            "border_style": "solid",  # solid, dashed, double, heavy, round
            "accent": "#007acc",
            "muted": "#595959",
            "panel_border": "blue",
        },
    )
    protocol: str = "openai"  # "openai" | "anthropic"
    tui: bool = True
    # False = let the terminal handle the mouse (native selection).
    # Default off on Windows: Textual's mouse-capture path is crash-prone
    # in PowerShell/conhost; native terminal mouse still allows selection.
    tui_mouse: bool = field(default_factory=lambda: os.name != "nt")
    privacy_mode: bool = (
        False  # When True, all data stays local, no cloud capabilities
    )
    debug: bool = (
        False  # When True, log final LLM prompts to .acli/logs/llm.log
    )
    skill_registry: str = (
        ""  # Optional registry index URL/path for /skill search/install
    )
    fallback_providers: list[str] = field(
        default_factory=list,
    )  # Ordered fallback provider names
    examples_repo: str = (
        ""  # Set to a public git URL to enable `acli example download`
    )
    examples_branch: str = "main"

    @property
    def api_key(self) -> str:
        """Return the API key for the currently selected provider."""
        return getattr(self, f"{self.provider}_api_key", "")

    @api_key.setter
    def api_key(self, value: str) -> None:
        """Assign the API key to the currently selected provider's slot."""
        setattr(self, f"{self.provider}_api_key", value)

    @classmethod
    def load(
        cls,
        default_provider: str = "tongyi",
        default_model: str = "qwen3.7-plus",
    ) -> Config:
        config = cls()
        # Caller's defaults become the initial values; config files and env
        # vars override them during the loading process below.
        config.provider = default_provider
        config.model = default_model

        # Load global config first (provider + api_key)
        if CONFIG_FILE.exists():
            config._load_global()

        # Load workspace config (project settings)
        workspace_loaded = False
        if WORKSPACE_CONFIG_FILE.exists():
            config._load_workspace()
            workspace_loaded = True
        elif CONFIG_FILE.exists():
            # Fallback: read workspace-level fields from global config
            # (migration)
            config._load_workspace_from(CONFIG_FILE)
            workspace_loaded = True

        # Environment variables fill in MISSING API keys (saved config takes
        # priority).
        if not config.tongyi_api_key:
            if key := os.environ.get("DASHSCOPE_API_KEY"):
                config.tongyi_api_key = key
        if not config.anthropic_api_key:
            if key := os.environ.get("ANTHROPIC_API_KEY"):
                config.anthropic_api_key = key
        if not config.openai_api_key:
            if key := os.environ.get("OPENAI_API_KEY"):
                config.openai_api_key = key

        # First-run provider auto-selection: if there is no saved workspace
        # config, pick a provider based on whichever API key env var is
        # present. Once the user has saved workspace config (e.g. via
        # /provider), that choice wins.
        if not workspace_loaded:
            if os.environ.get("DASHSCOPE_API_KEY"):
                config.provider = "tongyi"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                config.provider = "anthropic"
                config.model = "claude-sonnet-4-20250514"
            elif os.environ.get("OPENAI_API_KEY"):
                config.provider = "openai"
                config.model = "gpt-4o"

        # ACLI_PROVIDER / ACLI_MODEL always override (explicit user intent).
        if base_url := os.environ.get("OPENAI_BASE_URL"):
            config.base_url = base_url
        if model := os.environ.get("ACLI_MODEL"):
            config.model = model
        if provider := os.environ.get("ACLI_PROVIDER"):
            config.provider = provider
        if mcp_env := os.environ.get("ACLI_MCP_SERVERS"):
            for svc in mcp_env.split(","):
                svc = svc.strip()
                if svc:
                    config.mcp_servers.append(MCPServerConfig(service=svc))
        if user_id := os.environ.get("ACLI_MEMORY_USER_ID"):
            config.memory_user_id = user_id
        if lib_id := os.environ.get("ACLI_MEMORY_LIBRARY_ID"):
            config.memory_library_id = lib_id
        if os.environ.get("ACLI_MEMORY_DISABLED"):
            config.memory_enabled = False

        if config.provider == "dashscope":
            config.provider = "tongyi"

        # Validate: if the loaded model doesn't belong to the current provider,
        # fall back to the provider's first model. This handles the case where
        # a workspace config saves model = "qwen3.7-max" (tongyi) but the
        # provider later switches to anthropic, leaving a stale model name.
        # Skip validation when a custom backend path is set: the user is
        # responsible for the model name on their own endpoint.
        if not config.base_url:
            valid_models = _valid_models_for_provider(
                config.provider,
                config.custom_models,
            )
            if valid_models and config.model not in set(valid_models):
                config.model = valid_models[0]

        # Migrate session.json -> session/default/history.json
        old_session = WORKSPACE_DIR / "session.json"
        if old_session.exists():
            new_session = (
                WORKSPACE_DIR / "session" / "default" / "history.json"
            )
            new_session.parent.mkdir(parents=True, exist_ok=True)
            try:
                old_session.replace(new_session)
            except OSError:
                pass

        # Migrate session/history.json -> session/default/history.json
        legacy_session = WORKSPACE_DIR / "session" / "history.json"
        if legacy_session.exists():
            new_session = (
                WORKSPACE_DIR / "session" / "default" / "history.json"
            )
            new_session.parent.mkdir(parents=True, exist_ok=True)
            try:
                legacy_session.replace(new_session)
            except OSError:
                pass

        # Migrate history -> session/default/input-history.txt
        old_history = WORKSPACE_DIR / "history"
        if old_history.exists():
            new_history = (
                WORKSPACE_DIR / "session" / "default" / "input-history.txt"
            )
            new_history.parent.mkdir(parents=True, exist_ok=True)
            try:
                old_history.replace(new_history)
            except OSError:
                pass

        # Migrate session/input-history -> session/default/input-history.txt
        legacy_input = WORKSPACE_DIR / "session" / "input-history"
        if legacy_input.exists():
            new_input = (
                WORKSPACE_DIR / "session" / "default" / "input-history.txt"
            )
            new_input.parent.mkdir(parents=True, exist_ok=True)
            try:
                legacy_input.replace(new_input)
            except OSError:
                pass

        return config

    def _load_global(self):
        data = load_toml(CONFIG_FILE)
        if not data:
            return
        # Load per-provider API keys (with decryption)
        built_in_key_fields = {f"{prov}_api_key" for prov in PROVIDERS}
        for prov in PROVIDERS:
            key_field = f"{prov}_api_key"
            if key_field in data and not getattr(self, key_field):
                setattr(self, key_field, decrypt_value(str(data[key_field])))
        # Extension providers (ideatalk/deepseek/zhipu/...) may also store keys
        # as <name>_api_key in the global config file.
        for key, val in data.items():
            if (
                key.endswith("_api_key")
                and key not in built_in_key_fields
                and not getattr(self, key, "")
            ):
                setattr(self, key, decrypt_value(str(val)))
        # Legacy single api_key — migrate into current provider's slot
        if "api_key" in data and not self.api_key:
            self.api_key = decrypt_value(str(data["api_key"]))
        # Voice settings
        if "voice_silence_duration" in data:
            try:
                self.voice_silence_duration = float(
                    data["voice_silence_duration"],
                )
            except (ValueError, TypeError):
                pass
        if "voice_max_seconds" in data:
            try:
                self.voice_max_seconds = int(data["voice_max_seconds"])
            except (ValueError, TypeError):
                pass
        if "voice_silence_threshold" in data:
            try:
                self.voice_silence_threshold = int(
                    data["voice_silence_threshold"],
                )
            except (ValueError, TypeError):
                pass
        # TTS settings
        if "tts_enabled" in data:
            val = str(data["tts_enabled"]).lower()
            self.tts_enabled = val not in ("false", "0", "no")
        if "tts_model" in data:
            self.tts_model = str(data["tts_model"])
        if "tts_voice" in data:
            self.tts_voice = str(data["tts_voice"])
        if "tts_speed" in data:
            try:
                self.tts_speed = float(data["tts_speed"])
            except (ValueError, TypeError):
                pass
        if "examples_repo" in data:
            self.examples_repo = str(data["examples_repo"])
        if "examples_branch" in data:
            self.examples_branch = str(data["examples_branch"])

    def _load_workspace(self):
        self._load_workspace_from(WORKSPACE_CONFIG_FILE)

    def _load_workspace_from(self, path: Path):
        data = load_toml(path)
        if not data:
            return

        # Simple scalar fields
        if "provider" in data and not os.environ.get("ACLI_PROVIDER"):
            self.provider = str(data["provider"])
        if "model" in data and not os.environ.get("ACLI_MODEL"):
            self.model = str(data["model"])
        if "base_url" in data and not self.base_url:
            self.base_url = str(data["base_url"])
        if "memory_enabled" in data:
            val = str(data["memory_enabled"]).lower()
            self.memory_enabled = val not in ("false", "0", "no")
        if "memory_user_id" in data and not os.environ.get(
            "ACLI_MEMORY_USER_ID",
        ):
            self.memory_user_id = str(data["memory_user_id"])
        if "memory_library_id" in data and not os.environ.get(
            "ACLI_MEMORY_LIBRARY_ID",
        ):
            self.memory_library_id = str(data["memory_library_id"])
        if "user_name" in data and not self.user_name:
            self.user_name = str(data["user_name"])
        if "max_turns" in data:
            try:
                self.max_turns = int(data["max_turns"])
            except (ValueError, TypeError):
                pass
        if "session_persist" in data:
            val = str(data["session_persist"]).lower()
            self.session_persist = val not in ("false", "0", "no")
        if "tui" in data:
            val = str(data["tui"]).lower()
            self.tui = val not in ("false", "0", "no")
        if "tui_mouse" in data:
            val = str(data["tui_mouse"]).lower()
            self.tui_mouse = val not in ("false", "0", "no")
        if "privacy_mode" in data:
            val = str(data["privacy_mode"]).lower()
            self.privacy_mode = val not in ("false", "0", "no")
        if "debug" in data:
            val = str(data["debug"]).lower()
            self.debug = val not in ("false", "0", "no")
        if "voice_silence_duration" in data:
            try:
                self.voice_silence_duration = float(
                    data["voice_silence_duration"],
                )
            except (ValueError, TypeError):
                pass
        if "voice_max_seconds" in data:
            try:
                self.voice_max_seconds = int(data["voice_max_seconds"])
            except (ValueError, TypeError):
                pass
        if "voice_silence_threshold" in data:
            try:
                self.voice_silence_threshold = int(
                    data["voice_silence_threshold"],
                )
            except (ValueError, TypeError):
                pass
        if "tts_enabled" in data:
            val = str(data["tts_enabled"]).lower()
            self.tts_enabled = val not in ("false", "0", "no")
        if "tts_model" in data:
            self.tts_model = str(data["tts_model"])
        if "tts_voice" in data:
            self.tts_voice = str(data["tts_voice"])
        if "tts_speed" in data:
            try:
                self.tts_speed = float(data["tts_speed"])
            except (ValueError, TypeError):
                pass
        if "skill_registry" in data:
            self.skill_registry = str(data["skill_registry"])
        if "protocol" in data:
            self.protocol = str(data["protocol"])

        # API key (legacy - only if no env vars)
        if "api_key" in data:
            if (
                not any(
                    [
                        os.environ.get("DASHSCOPE_API_KEY"),
                        os.environ.get("ANTHROPIC_API_KEY"),
                        os.environ.get("OPENAI_API_KEY"),
                    ],
                )
                and not self.api_key
            ):
                self.api_key = decrypt_value(str(data["api_key"]))

        # List fields (comma-separated strings)
        if "enabled_capabilities" in data:
            raw = str(data["enabled_capabilities"])
            self.enabled_capabilities = [
                c.strip() for c in raw.split(",") if c.strip()
            ]
        if "custom_models" in data and not self.custom_models:
            raw = str(data["custom_models"])
            self.custom_models = [
                c.strip() for c in raw.split(",") if c.strip()
            ]
        if "fallback_providers" in data:
            raw = str(data["fallback_providers"])
            self.fallback_providers = [
                c.strip() for c in raw.split(",") if c.strip()
            ]

        # user_directives (TOML array)
        if "user_directives" in data and not self.user_directives:
            directives = data["user_directives"]
            if isinstance(directives, list):
                self.user_directives = [str(d) for d in directives if d]
        # theme (inline table or regular table)
        if "theme" in data:
            theme_data = data["theme"]
            if isinstance(theme_data, dict):
                self.theme = {str(k): str(v) for k, v in theme_data.items()}
            elif isinstance(theme_data, str):
                try:
                    parsed = parse_toml_inline_table(theme_data)
                    if parsed is not None:
                        self.theme = parsed
                except (ValueError, TypeError):
                    pass

        # delegation (inline table or regular table)
        if "delegation" in data:
            del_data = data["delegation"]
            if isinstance(del_data, dict):
                self.delegation = DelegationConfig.from_dict(del_data)
            elif isinstance(del_data, str):
                try:
                    parsed = parse_toml_inline_table(del_data)
                    if parsed is not None:
                        self.delegation = DelegationConfig.from_dict(parsed)
                except (ValueError, TypeError):
                    pass

        # subagents (array of tables)
        if "subagents" in data:
            for agent_data in data["subagents"]:
                if isinstance(agent_data, dict) and "key" in agent_data:
                    key = str(agent_data["key"])
                    self.subagents[key] = SubagentConfig.from_dict(agent_data)

        # mcp_servers (array of tables)
        if "mcp_servers" in data:
            for mcp_data in data["mcp_servers"]:
                if isinstance(mcp_data, dict) and "service" in mcp_data:
                    self.mcp_servers.append(MCPServerConfig(**mcp_data))
        if "examples_repo" in data:
            self.examples_repo = str(data["examples_repo"])
        if "examples_branch" in data:
            self.examples_branch = str(data["examples_branch"])
        if "thinking_provider" in data:
            self.thinking_provider = str(data["thinking_provider"])
        if "thinking_model" in data:
            self.thinking_model = str(data["thinking_model"])
        if "loop_mode" in data:
            self.loop_mode = str(data["loop_mode"])

    def save(self):
        if CONFIG_FILE == WORKSPACE_CONFIG_FILE:
            # cwd == ~: both scopes share one file; write keys + settings
            # together
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                CONFIG_FILE,
                "\n".join(self._global_lines() + self._workspace_lines())
                + "\n",
            )
            return
        self.save_global()
        self.save_workspace()

    def save_global(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(CONFIG_FILE, "\n".join(self._global_lines()) + "\n")

    def _global_lines(self) -> list[str]:
        lines = []
        built_in_key_fields = {f"{prov}_api_key" for prov in PROVIDERS}
        for prov in PROVIDERS:
            key_val = getattr(self, f"{prov}_api_key", "")
            if key_val:
                lines.append(
                    f"{prov}_api_key = {toml_str(encrypt_value(key_val))}",
                )
        # Extension provider keys stored as <name>_api_key (e.g.
        # ideatalk_api_key)
        for attr in self.__dict__:
            if attr.endswith("_api_key") and attr not in built_in_key_fields:
                key_val = getattr(self, attr, "")
                if key_val:
                    lines.append(
                        f"{attr} = {toml_str(encrypt_value(key_val))}",
                    )
        if self.tts_enabled:
            lines.append("tts_enabled = true")
        if self.tts_model and self.tts_model != "cosyvoice-v2":
            lines.append(f"tts_model = {toml_str(self.tts_model)}")
        if self.tts_voice and self.tts_voice != "longxiaochun_v2":
            lines.append(f"tts_voice = {toml_str(self.tts_voice)}")
        if self.tts_speed != 1.0:
            lines.append(f"tts_speed = {self.tts_speed}")
        if self.voice_silence_duration != 2.0:
            lines.append(
                f"voice_silence_duration = {self.voice_silence_duration}",
            )
        if self.voice_max_seconds != 60:
            lines.append(f"voice_max_seconds = {self.voice_max_seconds}")
        if self.voice_silence_threshold != 500:
            lines.append(
                f"voice_silence_threshold = {self.voice_silence_threshold}",
            )
        return lines

    def save_workspace(self):
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            WORKSPACE_CONFIG_FILE,
            "\n".join(self._workspace_lines()) + "\n",
        )

    def _workspace_lines(self) -> list[str]:
        lines = []
        if self.provider:
            lines.append(f"provider = {toml_str(self.provider)}")
        if self.user_name:
            lines.append(f"user_name = {toml_str(self.user_name)}")
        if self.model:
            lines.append(f"model = {toml_str(self.model)}")
        if self.base_url:
            lines.append(f"base_url = {toml_str(self.base_url)}")
        if self.enabled_capabilities is not None:
            lines.append(
                f"enabled_capabilities = "
                f"{toml_str(','.join(self.enabled_capabilities))}",
            )
        if self.custom_models:
            lines.append(
                f"custom_models = {toml_str(','.join(self.custom_models))}",
            )
        if self.max_turns != 50:
            lines.append(f"max_turns = {self.max_turns}")
        if not self.session_persist:
            lines.append("session_persist = false")
        if self.user_directives:
            directives = ", ".join(toml_str(d) for d in self.user_directives)
            lines.append(f"user_directives = [{directives}]")
        if not self.memory_enabled:
            lines.append("memory_enabled = false")
        if self.memory_user_id:
            lines.append(f"memory_user_id = {toml_str(self.memory_user_id)}")
        if self.memory_library_id:
            lines.append(
                f"memory_library_id = {toml_str(self.memory_library_id)}",
            )
        if self.theme:
            pairs = ", ".join(
                f"{toml_str(k)} = {toml_str(v)}" for k, v in self.theme.items()
            )
            lines.append(f"theme = {{{pairs}}}")
        if self.delegation != DelegationConfig():
            d = self.delegation
            pairs = (
                f'"max_concurrent" = {d.max_concurrent}, '
                f'"default_timeout" = {d.default_timeout}, '
                f'"allow_nested" = {str(d.allow_nested).lower()}'
            )
            lines.append(f"delegation = {{{pairs}}}")
        if self.protocol and self.protocol != "openai":
            lines.append(f"protocol = {toml_str(self.protocol)}")
        if self.fallback_providers:
            lines.append(
                f"fallback_providers = "
                f"{toml_str(','.join(self.fallback_providers))}",
            )
        if self.skill_registry:
            lines.append(f"skill_registry = {toml_str(self.skill_registry)}")
        if not self.tui:
            lines.append("tui = false")
        if not self.tui_mouse:
            lines.append("tui_mouse = false")
        if self.privacy_mode:
            lines.append("privacy_mode = true")
        if self.debug:
            lines.append("debug = true")
        lines.append(f"tts_enabled = {str(self.tts_enabled).lower()}")
        if self.tts_model and self.tts_model != "cosyvoice-v2":
            lines.append(f"tts_model = {toml_str(self.tts_model)}")
        if self.tts_voice and self.tts_voice != "longxiaochun_v2":
            lines.append(f"tts_voice = {toml_str(self.tts_voice)}")
        if self.tts_speed != 1.0:
            lines.append(f"tts_speed = {self.tts_speed}")
        if self.voice_silence_duration != 2.0:
            lines.append(
                f"voice_silence_duration = {self.voice_silence_duration}",
            )
        if self.voice_max_seconds != 60:
            lines.append(f"voice_max_seconds = {self.voice_max_seconds}")
        if self.voice_silence_threshold != 500:
            lines.append(
                f"voice_silence_threshold = {self.voice_silence_threshold}",
            )
        if self.examples_repo:
            lines.append(f"examples_repo = {toml_str(self.examples_repo)}")
        if self.examples_branch != "main":
            lines.append(f"examples_branch = {toml_str(self.examples_branch)}")
        if self.thinking_provider:
            lines.append(
                f"thinking_provider = {toml_str(self.thinking_provider)}",
            )
        if self.thinking_model:
            lines.append(f"thinking_model = {toml_str(self.thinking_model)}")
        if self.loop_mode and self.loop_mode != "auto":
            lines.append(f"loop_mode = {toml_str(self.loop_mode)}")
        # Array-of-tables sections must come LAST: in TOML, bare scalar keys
        # written after a [[table]] header belong to that table, so any scalar
        # appended here would silently move into the last subagent/mcp entry.
        for key, ac in self.subagents.items():
            lines.append("")
            lines.append("[[subagents]]")
            lines.append(f"key = {toml_str(key)}")
            if ac.model:
                lines.append(f"model = {toml_str(ac.model)}")
            if ac.temperature:
                lines.append(f"temperature = {ac.temperature}")
            if ac.max_turns:
                lines.append(f"max_turns = {ac.max_turns}")
        for mcp in self.mcp_servers:
            lines.append("")
            lines.append("[[mcp_servers]]")
            lines.append(f"service = {toml_str(mcp.service)}")
            if mcp.url:
                lines.append(f"url = {toml_str(mcp.url)}")
        return lines
