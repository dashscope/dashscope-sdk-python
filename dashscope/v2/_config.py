# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Configuration resolution for the dashscope 2.0 SDK.

Priority: explicit parameters > environment variables
> ~/.dashscope/config.toml. For ``api_key`` only there is one more,
lowest tier: the 1.x key file (``DASHSCOPE_API_KEY_FILE_PATH`` or
``~/.dashscope/api_key``) written by ``save_api_key``, so 1.x users who
stored their key that way keep working.

When no base_url is configured at any level, every default endpoint is
built from the workspace-scoped MaaS gateway
(``https://{workspace}.{region}.maas.aliyuncs.com``), which serves each
protocol under its own path. ``region`` (default ``cn-beijing``) selects
one of the six regional gateways, so international users pass
``region=...`` instead of hand-building a ``base_url``. When no
``workspace`` is available either, resolution falls back to the legacy
shared gateway
(``https://dashscope.aliyuncs.com``, the 1.x endpoint -- still supported
server-side, though the workspace-scoped gateway is recommended) and
emits a ``UserWarning`` nudging users to configure a workspace, so an
api_key-only quickstart keeps working. The protocol-dependent default
endpoint (``openai`` / ``anthropic`` compatible endpoints) applies only
to ``protocol_base_url``, which the protocol-adapted chat family
(``chat.completions`` / ``multimodal.chat``) sends its requests to; every
other resource always talks to ``base_url``. Setting an explicit
``base_url`` remains an escape hatch for custom/private gateways and does
not require a ``workspace``.
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

# Workspace-scoped MaaS gateway: the recommended endpoint host, used for
# every default route when a workspace is configured. Without a workspace
# (and no explicit base_url), resolution falls back to the legacy shared
# gateway below.
WORKSPACE_GATEWAY_TEMPLATE = "https://{workspace}.{region}.maas.aliyuncs.com"

# The MaaS gateway is deployed in six regions; ``region`` picks which one
# the default endpoints point at. cn-beijing is the default so existing
# configurations keep their URLs unchanged.
DEFAULT_REGION = "cn-beijing"
VALID_REGIONS = (
    "cn-beijing",
    "ap-southeast-1",
    "cn-hongkong",
    "us-east-1",
    "eu-central-1",
    "ap-northeast-1",
)

# Legacy shared gateways (the 1.x endpoints), per region. Still supported
# server-side with no announced retirement; v2 uses them only as the
# no-workspace fallback so an api_key-only quickstart keeps working, and
# warns the user toward the workspace-scoped gateway. The wire-protocol
# paths are identical on both gateways; only the host differs. Regions
# without a dedicated shared gateway use the international one.
LEGACY_GATEWAY_HOSTS = {
    "cn-beijing": "https://dashscope.aliyuncs.com",
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com",
    "cn-hongkong": "https://cn-hongkong.dashscope.aliyuncs.com",
    "us-east-1": "https://dashscope-intl.aliyuncs.com",
    "eu-central-1": "https://dashscope-intl.aliyuncs.com",
    "ap-northeast-1": "https://dashscope-intl.aliyuncs.com",
}
DEFAULT_PROTOCOL = "native"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 2
VALID_PROTOCOLS = ("native", "openai", "anthropic")

# HTTP keep-alive pool knobs (matching the httpx defaults). One Client
# holds one pool shared by every resource call on it, so connections
# stay warm across requests -- the 1.x SDK built a fresh Session per
# call instead.
DEFAULT_HTTP_MAX_CONNECTIONS = 100
DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS = 20
DEFAULT_HTTP_KEEPALIVE_EXPIRY = 5.0

# Each wire protocol lives at its own URL path under the gateway host.
PROTOCOL_DEFAULT_PATHS = {
    "native": "/api/v1",
    "openai": "/compatible-mode/v1",
    "anthropic": "/apps/anthropic",
}

# The MaaS gateway serves the websocket endpoints under the same host as
# HTTP (only the scheme differs); the legacy shared gateway below is the
# no-workspace fallback.
WORKSPACE_WS_GATEWAY_TEMPLATE = "wss://{workspace}.{region}.maas.aliyuncs.com"
LEGACY_WS_GATEWAY_HOSTS = {
    "cn-beijing": "wss://dashscope.aliyuncs.com",
    "ap-southeast-1": "wss://dashscope-intl.aliyuncs.com",
    "cn-hongkong": "wss://cn-hongkong.dashscope.aliyuncs.com",
    "us-east-1": "wss://dashscope-intl.aliyuncs.com",
    "eu-central-1": "wss://dashscope-intl.aliyuncs.com",
    "ap-northeast-1": "wss://dashscope-intl.aliyuncs.com",
}
WS_INFERENCE_PATH = "/api-ws/v1/inference"
WS_REALTIME_PATH = "/api-ws/v1/realtime"

_LEGACY_GATEWAY_FALLBACK_MSG = (
    "no workspace configured: falling back to the legacy shared gateway "
    "({}). For better performance and stability, configure a workspace "
    "to use the recommended workspace-scoped gateway: set the "
    "DASHSCOPE_WORKSPACE environment variable, pass workspace=..., or "
    "add 'workspace' to your config.toml."
)

_API_KEY_ENV = "DASHSCOPE_API_KEY"
_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
# 1.x env var name, honored as fallback so existing setups keep working
_LEGACY_BASE_URL_ENV = "DASHSCOPE_HTTP_BASE_URL"
_WORKSPACE_ENV = "DASHSCOPE_WORKSPACE"
_REGION_ENV = "DASHSCOPE_REGION"
_WS_BASE_URL_ENV = "DASHSCOPE_WEBSOCKET_BASE_URL"
_PROTOCOL_ENV = "DASHSCOPE_PROTOCOL"
_TIMEOUT_ENV = "DASHSCOPE_TIMEOUT"
_MAX_RETRIES_ENV = "DASHSCOPE_MAX_RETRIES"
_HTTP_MAX_CONNECTIONS_ENV = "DASHSCOPE_HTTP_MAX_CONNECTIONS"
_HTTP_MAX_KEEPALIVE_CONNECTIONS_ENV = (
    "DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS"
)
_HTTP_KEEPALIVE_EXPIRY_ENV = "DASHSCOPE_HTTP_KEEPALIVE_EXPIRY"
_LOG_ENV = "DASHSCOPE_LOG"
_CONFIG_PATH_ENV = "DASHSCOPE_CONFIG"
_DEFAULT_CONFIG_PATH = "~/.dashscope/config.toml"
# 1.x key-file auth (dashscope.save_api_key): env-specified path first,
# then the default file -- the lowest api_key tier.
_API_KEY_FILE_PATH_ENV = "DASHSCOPE_API_KEY_FILE_PATH"
_DEFAULT_API_KEY_FILE_PATH = "~/.dashscope/api_key"

_FILE_KEYS = frozenset(
    {
        "api_key",
        "base_url",
        "workspace",
        "region",
        "protocol",
        "timeout",
        "max_retries",
        "http_max_connections",
        "http_max_keepalive_connections",
        "http_keepalive_expiry",
    },
)


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"invalid {name}={raw!r}: expected a number",
        ) from None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"invalid {name}={raw!r}: expected an integer",
        ) from None


def _legacy_fallback(host: str) -> str:
    """Return the legacy gateway host, nudging toward a workspace.

    A single warn site keeps the warning once-per-process under the
    default warnings filter.
    """
    warnings.warn(
        _LEGACY_GATEWAY_FALLBACK_MSG.format(host),
        UserWarning,
        stacklevel=2,
    )
    return host


def _validate_region(region: str) -> str:
    if region not in VALID_REGIONS:
        raise ValueError(
            f"invalid region {region!r}, expected one of {VALID_REGIONS}",
        )
    return region


def _protocol_default_base_url(
    protocol: str,
    workspace: Optional[str],
    region: Optional[str] = None,
) -> str:
    region = _validate_region(region or DEFAULT_REGION)
    if workspace:
        host = WORKSPACE_GATEWAY_TEMPLATE.format(
            workspace=workspace,
            region=region,
        )
    else:
        host = _legacy_fallback(LEGACY_GATEWAY_HOSTS[region])
    return host + PROTOCOL_DEFAULT_PATHS[protocol]


def compatible_mode_base_url(
    workspace: Optional[str],
    region: Optional[str] = None,
    configured_base_url: Optional[str] = None,
) -> str:
    """Endpoint root for services mounted under the compatible-mode path.

    The System One decision model (``/systemone``) is served under
    ``/compatible-mode/v1`` on every gateway, regardless of the client's
    chat ``protocol`` setting, so its base can come from neither
    ``Config.base_url`` (native ``/api/v1``) nor
    ``Config.protocol_base_url`` (which follows the configured chat
    protocol). An explicit base_url stays the service root -- the same
    rule ``protocol_base_url`` applies; otherwise the default gateway
    host (workspace-scoped, or the legacy shared fallback) carries the
    fixed compatible-mode path.
    """
    if configured_base_url is not None:
        return configured_base_url
    return _protocol_default_base_url("openai", workspace, region)


def workspace_ws_url(
    workspace: Optional[str],
    path: str,
    region: Optional[str] = None,
) -> str:
    """Build the default websocket endpoint for ``path``.

    The MaaS gateway serves websocket under the same host as HTTP, so the
    URL is ``wss://{workspace}.{region}.maas.aliyuncs.com{path}``.
    Without a workspace it falls back to the region's legacy shared
    gateway (``wss://dashscope.aliyuncs.com`` for cn-beijing) with a
    warning.
    """
    region = _validate_region(region or DEFAULT_REGION)
    if workspace:
        host = WORKSPACE_WS_GATEWAY_TEMPLATE.format(
            workspace=workspace,
            region=region,
        )
    else:
        host = _legacy_fallback(LEGACY_WS_GATEWAY_HOSTS[region])
    return host + path


@dataclass(frozen=True)
class Config:
    api_key: Optional[str]
    base_url: str
    workspace: Optional[str]
    protocol: str
    timeout: float
    max_retries: int
    # Where the protocol-adapted chat family (chat/multimodal) sends
    # requests: the configured base_url if any, else the protocol's
    # default endpoint.
    protocol_base_url: str
    # The configured base_url (any level) before defaults kicked in;
    # None means every resolved URL is a default. Lets with_options()
    # re-resolve per-protocol defaults for a derived protocol.
    configured_base_url: Optional[str] = None
    # The region the default gateways were built for. Informational only
    # when an explicit base_url overrode them.
    region: str = DEFAULT_REGION
    # HTTP connection-pool sizes for the client's httpx client.
    http_max_connections: int = DEFAULT_HTTP_MAX_CONNECTIONS
    http_max_keepalive_connections: int = (
        DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS
    )
    http_keepalive_expiry: float = DEFAULT_HTTP_KEEPALIVE_EXPIRY


class Settings:
    """Global mutable settings facade (design doc M3 3.11).

    Single runtime entry point: values assigned here override environment
    variables and ``config.toml`` for every Client created afterwards.
    Unset properties fall back lazily to the environment (read at access
    time, not import time).
    """

    def __init__(self) -> None:
        self._explicit: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_explicit(self, name: str) -> Any:
        """Explicitly assigned value, or None if unset."""
        with self._lock:
            return self._explicit.get(name)

    def clear(self) -> None:
        """Forget every explicit assignment (back to env / defaults).

        If ``log_level`` had been assigned, the logger level is reset to
        the ``DASHSCOPE_LOG`` env level (or WARNING) so it does not leak.
        """
        with self._lock:
            had_log_level = "log_level" in self._explicit
            self._explicit.clear()
        if had_log_level:
            # pylint: disable=import-outside-toplevel
            from dashscope.v2._logging import reset_log_level

            reset_log_level()

    @property
    def api_key(self) -> Optional[str]:
        return self._explicit.get("api_key") or os.environ.get(_API_KEY_ENV)

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        with self._lock:
            self._explicit["api_key"] = value

    @property
    def base_url(self) -> str:
        return (
            self._explicit.get("base_url")
            or os.environ.get(_BASE_URL_ENV)
            or os.environ.get(_LEGACY_BASE_URL_ENV)
            or _protocol_default_base_url(
                "native",
                self.workspace,
                self.region,
            )
        )

    @base_url.setter
    def base_url(self, value: Optional[str]) -> None:
        with self._lock:
            self._explicit["base_url"] = value

    @property
    def websocket_base_url(self) -> Optional[str]:
        return self._explicit.get(
            "websocket_base_url",
        ) or os.environ.get(_WS_BASE_URL_ENV)

    @websocket_base_url.setter
    def websocket_base_url(self, value: Optional[str]) -> None:
        with self._lock:
            self._explicit["websocket_base_url"] = value

    @property
    def workspace(self) -> Optional[str]:
        return self._explicit.get("workspace") or os.environ.get(
            _WORKSPACE_ENV,
        )

    @workspace.setter
    def workspace(self, value: Optional[str]) -> None:
        with self._lock:
            self._explicit["workspace"] = value

    @property
    def region(self) -> str:
        return (
            self._explicit.get("region")
            or os.environ.get(_REGION_ENV)
            or DEFAULT_REGION
        )

    @region.setter
    def region(self, value: str) -> None:
        _validate_region(value)
        with self._lock:
            self._explicit["region"] = value

    @property
    def protocol(self) -> str:
        return (
            self._explicit.get("protocol")
            or os.environ.get(_PROTOCOL_ENV)
            or DEFAULT_PROTOCOL
        )

    @protocol.setter
    def protocol(self, value: str) -> None:
        if value not in VALID_PROTOCOLS:
            raise ValueError(
                f"invalid protocol {value!r}, "
                f"expected one of {VALID_PROTOCOLS}",
            )
        with self._lock:
            self._explicit["protocol"] = value

    @property
    def timeout(self) -> float:
        value = self._explicit.get("timeout")
        if value is None:
            value = _env_float(_TIMEOUT_ENV)
        return DEFAULT_TIMEOUT if value is None else value

    @timeout.setter
    def timeout(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"timeout must be positive, got {value}")
        with self._lock:
            self._explicit["timeout"] = value

    @property
    def max_retries(self) -> int:
        value = self._explicit.get("max_retries")
        if value is None:
            value = _env_int(_MAX_RETRIES_ENV)
        return DEFAULT_MAX_RETRIES if value is None else value

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError(f"max_retries must be >= 0, got {value}")
        with self._lock:
            self._explicit["max_retries"] = value

    @property
    def http_max_connections(self) -> int:
        value = self._explicit.get("http_max_connections")
        if value is None:
            value = _env_int(_HTTP_MAX_CONNECTIONS_ENV)
        return DEFAULT_HTTP_MAX_CONNECTIONS if value is None else value

    @http_max_connections.setter
    def http_max_connections(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError(
                f"http_max_connections must be >= 1, got {value}",
            )
        with self._lock:
            self._explicit["http_max_connections"] = value

    @property
    def http_max_keepalive_connections(self) -> int:
        value = self._explicit.get("http_max_keepalive_connections")
        if value is None:
            value = _env_int(_HTTP_MAX_KEEPALIVE_CONNECTIONS_ENV)
        return (
            DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS if value is None else value
        )

    @http_max_keepalive_connections.setter
    def http_max_keepalive_connections(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError(
                f"http_max_keepalive_connections must be >= 0, got {value}",
            )
        with self._lock:
            self._explicit["http_max_keepalive_connections"] = value

    @property
    def http_keepalive_expiry(self) -> float:
        value = self._explicit.get("http_keepalive_expiry")
        if value is None:
            value = _env_float(_HTTP_KEEPALIVE_EXPIRY_ENV)
        return DEFAULT_HTTP_KEEPALIVE_EXPIRY if value is None else value

    @http_keepalive_expiry.setter
    def http_keepalive_expiry(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(
                f"http_keepalive_expiry must be positive, got {value}",
            )
        with self._lock:
            self._explicit["http_keepalive_expiry"] = value

    @property
    def log_level(self) -> Optional[str]:
        return self._explicit.get("log_level") or os.environ.get(_LOG_ENV)

    @log_level.setter
    def log_level(self, value: str) -> None:
        # pylint: disable=import-outside-toplevel
        from dashscope.v2._logging import set_log_level

        set_log_level(value)
        with self._lock:
            self._explicit["log_level"] = value


settings = Settings()


def default_config_path() -> Path:
    return Path(
        os.environ.get(_CONFIG_PATH_ENV, _DEFAULT_CONFIG_PATH),
    ).expanduser()


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"invalid TOML in config file {path}: {e}") from e
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in _FILE_KEYS}


def _api_key_from_file() -> Optional[str]:
    """Lowest api_key tier: the 1.x key file.

    Same order as ``dashscope.common.api_key.get_default_api_key`` --
    ``DASHSCOPE_API_KEY_FILE_PATH`` first, then the default
    ``~/.dashscope/api_key`` written by ``save_api_key``. A missing or
    unreadable file resolves to None rather than raising, since this is
    only a fallback tier.
    """
    path = os.environ.get(_API_KEY_FILE_PATH_ENV) or _DEFAULT_API_KEY_FILE_PATH
    try:
        with open(os.path.expanduser(path), "rt", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _pick(
    explicit: Any,
    settings_value: Any,
    env_value: Any,
    file_value: Any,
    default: Any,
) -> Any:
    if explicit is not None:
        return explicit
    if settings_value is not None:
        return settings_value
    # Missing env vars parse to None and empty strings count as unset,
    # but a real 0 (e.g. keepalive=0, max_retries=0) is a value.
    if env_value is not None and env_value != "":
        return env_value
    if file_value is not None:
        return file_value
    return default


def resolve_config(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    workspace: Optional[str] = None,
    region: Optional[str] = None,
    protocol: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    config_path: Union[str, Path, None] = None,
) -> Config:
    path = (
        Path(config_path).expanduser()
        if config_path
        else default_config_path()
    )
    file_cfg = _load_config_file(path)

    resolved_protocol = str(
        _pick(
            protocol,
            settings.get_explicit("protocol"),
            os.environ.get(_PROTOCOL_ENV),
            file_cfg.get("protocol"),
            DEFAULT_PROTOCOL,
        ),
    )
    if resolved_protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"invalid protocol {resolved_protocol!r}, "
            f"expected one of {VALID_PROTOCOLS}",
        )

    resolved_timeout = float(
        _pick(
            timeout,
            settings.get_explicit("timeout"),
            _env_float(_TIMEOUT_ENV),
            file_cfg.get("timeout"),
            DEFAULT_TIMEOUT,
        ),
    )
    if resolved_timeout <= 0:
        raise ValueError(f"timeout must be positive, got {resolved_timeout}")

    resolved_max_retries = int(
        _pick(
            max_retries,
            settings.get_explicit("max_retries"),
            _env_int(_MAX_RETRIES_ENV),
            file_cfg.get("max_retries"),
            DEFAULT_MAX_RETRIES,
        ),
    )
    if resolved_max_retries < 0:
        raise ValueError(
            f"max_retries must be >= 0, got {resolved_max_retries}",
        )

    resolved_http_max_connections = int(
        _pick(
            None,
            settings.get_explicit("http_max_connections"),
            _env_int(_HTTP_MAX_CONNECTIONS_ENV),
            file_cfg.get("http_max_connections"),
            DEFAULT_HTTP_MAX_CONNECTIONS,
        ),
    )
    if resolved_http_max_connections < 1:
        raise ValueError(
            "http_max_connections must be >= 1, "
            f"got {resolved_http_max_connections}",
        )

    resolved_http_max_keepalive = int(
        _pick(
            None,
            settings.get_explicit("http_max_keepalive_connections"),
            _env_int(_HTTP_MAX_KEEPALIVE_CONNECTIONS_ENV),
            file_cfg.get("http_max_keepalive_connections"),
            DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )
    if resolved_http_max_keepalive < 0:
        raise ValueError(
            "http_max_keepalive_connections must be >= 0, "
            f"got {resolved_http_max_keepalive}",
        )

    resolved_http_keepalive_expiry = float(
        _pick(
            None,
            settings.get_explicit("http_keepalive_expiry"),
            _env_float(_HTTP_KEEPALIVE_EXPIRY_ENV),
            file_cfg.get("http_keepalive_expiry"),
            DEFAULT_HTTP_KEEPALIVE_EXPIRY,
        ),
    )
    if resolved_http_keepalive_expiry <= 0:
        raise ValueError(
            "http_keepalive_expiry must be positive, "
            f"got {resolved_http_keepalive_expiry}",
        )

    resolved_workspace = _pick(
        workspace,
        settings.get_explicit("workspace"),
        os.environ.get(_WORKSPACE_ENV),
        file_cfg.get("workspace"),
        None,
    )

    resolved_region = _validate_region(
        str(
            _pick(
                region,
                settings.get_explicit("region"),
                os.environ.get(_REGION_ENV),
                file_cfg.get("region"),
                DEFAULT_REGION,
            ),
        ),
    )

    configured_base_url = _pick(
        base_url,
        settings.get_explicit("base_url"),
        os.environ.get(_BASE_URL_ENV) or os.environ.get(_LEGACY_BASE_URL_ENV),
        file_cfg.get("base_url"),
        None,
    )
    # An explicit base_url is the escape hatch for custom/private gateways.
    # Otherwise defaults come from the workspace-scoped MaaS gateway,
    # falling back to the legacy shared gateway (with a warning) when no
    # workspace is configured (_protocol_default_base_url).
    if configured_base_url is not None:
        resolved_base_url = configured_base_url
        resolved_protocol_base_url = configured_base_url
    else:
        resolved_base_url = _protocol_default_base_url(
            "native",
            resolved_workspace,
            resolved_region,
        )
        resolved_protocol_base_url = _protocol_default_base_url(
            resolved_protocol,
            resolved_workspace,
            resolved_region,
        )

    return Config(
        api_key=_pick(
            api_key,
            settings.get_explicit("api_key"),
            os.environ.get(_API_KEY_ENV),
            file_cfg.get("api_key"),
            _api_key_from_file(),
        ),
        base_url=resolved_base_url,
        workspace=resolved_workspace,
        protocol=resolved_protocol,
        timeout=resolved_timeout,
        max_retries=resolved_max_retries,
        protocol_base_url=resolved_protocol_base_url,
        configured_base_url=configured_base_url,
        region=resolved_region,
        http_max_connections=resolved_http_max_connections,
        http_max_keepalive_connections=resolved_http_max_keepalive,
        http_keepalive_expiry=resolved_http_keepalive_expiry,
    )
