# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-invocation CLI state, hung on ``ctx.obj`` by the root callback.

click propagates ``ctx.obj`` down the command tree, so a command only
has to declare ``ctx: typer.Context`` and then reach for
``ctx.obj.client`` / ``ctx.obj.renderer``. Both are built lazily: a
command that errors out on its own arguments never constructs an SDK
client, and ``--help`` never reads the config file.

Precedence is the one the design doc asks for -- command line beats
environment beats config file -- and it is obtained for free rather than
reimplemented: everything the user spelled on the command line is handed
to ``Client(...)`` as an explicit keyword, which is the SDK's
highest-priority tier. The CLI never reads ``DASHSCOPE_*`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from dashscope.cli2 import config_store, sdk
from dashscope.cli2.errors import usage_error
from dashscope.cli2.output import TABLE, Renderer, resolve_no_color

# Global flags that map straight onto Client(...) keywords.
_FLAG_TO_SDK = {
    "api_key": "api_key",
    "protocol": "protocol",
    "base_url": "base_url",
    "workspace": "workspace",
    "region": "region",
    "timeout": "timeout",
}


@dataclass(frozen=True)
class CliDefaults:
    """CLI-owned defaults, read from the ``[cli]`` table of config.toml.

    The SDK's reader filters config keys through its own allowlist, which
    does not include these, so the two readers share one file without
    either seeing the other's keys.
    """

    default_model: Optional[str] = None
    output: Optional[str] = None


@dataclass(frozen=True)
class GlobalOptions:
    """Everything the root callback collected, before any resolution.

    Kept as a plain value object so tests can construct a context without
    running a Typer app.
    """

    api_key: Optional[str] = None
    protocol: Optional[str] = None
    base_url: Optional[str] = None
    workspace: Optional[str] = None
    region: Optional[str] = None
    timeout: Optional[float] = None
    output: Optional[str] = None
    # Raw ``-c KEY=VALUE`` strings, unparsed: resolution needs to know
    # whether a key belongs to the SDK or to the CLI, and that check
    # produces the error message.
    overrides: Tuple[str, ...] = ()
    no_color: bool = False
    verbose: bool = False


@dataclass
class CliContext:
    """Lazily-resolved view of one invocation."""

    options: GlobalOptions = field(default_factory=GlobalOptions)

    def __post_init__(self) -> None:
        self.verbose = self.options.verbose
        self._defaults: Optional[CliDefaults] = None
        self._cli_overrides: Optional[Dict[str, Any]] = None
        self._sdk_overrides: Optional[Dict[str, Any]] = None
        self._client: Any = None
        self._renderer: Optional[Renderer] = None

    # -- configuration ---------------------------------------------------

    def _resolve_overrides(self) -> None:
        """Split ``-c KEY=VALUE`` into SDK settings and CLI defaults.

        Runs once, on first access to either side, so an invalid key is
        reported before any network work.
        """
        sdk_keys = config_store._sdk_keys()  # pylint: disable=protected-access
        sdk_kw: Dict[str, Any] = {}
        cli_kw: Dict[str, Any] = {}
        for raw in self.options.overrides:
            key, sep, value = raw.partition("=")
            key = key.strip()
            if not sep:
                raise usage_error(
                    f"-c expects KEY=VALUE, got {raw!r}",
                )
            if not key:
                raise usage_error(f"-c expects KEY=VALUE, got {raw!r}")
            if key in sdk_keys:
                sdk_kw[key] = config_store.parse_toml_value(value)
            elif key in config_store.CLI_KEYS:
                cli_kw[key] = config_store.parse_toml_value(value)
            else:
                known = sorted(sdk_keys | config_store.CLI_KEYS)
                raise usage_error(
                    f"unknown configuration key {key!r}; "
                    f"known keys: {', '.join(known)}",
                )
        # Validate and coerce through the same path `config set` uses, so
        # `-c protocol=bogus` and `config set protocol bogus` fail alike,
        # and `-c timeout=30` becomes a number rather than "30".
        for key, value in sdk_kw.items():
            sdk_kw[
                key
            ] = config_store._validate(  # pylint: disable=protected-access
                key,
                value,
            )
        for key, value in cli_kw.items():
            cli_kw[
                key
            ] = config_store._validate(  # pylint: disable=protected-access
                key,
                value,
            )
        self._sdk_overrides = sdk_kw
        self._cli_overrides = cli_kw

    def _cli_overrides_map(self) -> Dict[str, Any]:
        if self._cli_overrides is None:
            self._resolve_overrides()
        return self._cli_overrides or {}

    @property
    def defaults(self) -> CliDefaults:
        """``[cli]`` values from the config file, overridden by ``-c``."""
        if self._defaults is None:
            stored = config_store.load_cli_section()
            merged = dict(stored)
            merged.update(self._cli_overrides_map())
            self._defaults = CliDefaults(
                default_model=merged.get("default_model"),
                output=merged.get("output"),
            )
        return self._defaults

    def sdk_overrides(self) -> Dict[str, Any]:
        """Explicit keywords for ``Client(...)``.

        A dedicated flag wins over ``-c`` for the same setting: the flag
        is the more specific spelling, and letting ``-c`` win would make
        ``dashscope -k sk-a -c api_key=sk-b`` silently drop ``sk-a``.
        """
        if self._sdk_overrides is None:
            self._resolve_overrides()
        kwargs = dict(self._sdk_overrides or {})
        for name, sdk_name in _FLAG_TO_SDK.items():
            value = getattr(self.options, name)
            if value is not None:
                kwargs[sdk_name] = value
        return kwargs

    # -- derived objects -------------------------------------------------

    @property
    def output_format(self) -> str:
        """``--output`` flag, else ``-c output=``, else ``[cli]``, else table."""
        return self.options.output or self.defaults.output or TABLE

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer(
                self.output_format,
                no_color=resolve_no_color(self.options.no_color),
                verbose=self.options.verbose,
            )
        return self._renderer

    @property
    def client(self) -> Any:
        """The SDK client, built on first use.

        Reached through the :mod:`dashscope.cli2.sdk` module attribute
        rather than a direct import so tests have one seam to patch.
        """
        if self._client is None:
            self._client = sdk.build_client(self.sdk_overrides())
        return self._client

    def close(self) -> None:
        """Release the client, if one was ever built.

        Called by the error handler too, so a failing command does not
        leak the httpx connection pool.
        """
        client, self._client = self._client, None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
