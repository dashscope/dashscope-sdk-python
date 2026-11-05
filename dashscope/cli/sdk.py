# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The single place the CLI imports the v2 SDK's :class:`Client`.

``import dashscope`` pulls in the 1.x resource tree and the pydantic
models -- hundreds of milliseconds that ``dashscope chat create --help``
should not pay. Every command therefore reaches the SDK through
:func:`build_client`, which runs only when a command actually calls the
service, and which tests monkeypatch in one place.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional

from dashscope.cli.errors import usage_error

# Marks CLI traffic on the User-Agent so the service can tell a request
# came from the command line rather than from user SDK code. This is a
# different mechanism from 1.x's `set_sdk_client()`, which only sets the
# `x-dashscope-sdk-client` header.
USER_AGENT_SUFFIX = "dashscope-cli"

_CLIENT_PARAMS: Optional[FrozenSet[str]] = None


def _client_params() -> FrozenSet[str]:
    """The keyword names ``Client.__init__`` actually accepts.

    Read out of the signature rather than listed here, because the set of
    ``config.toml`` keys is *not* the set of constructor arguments: the
    connection-pool trio is resolved from the file and the environment
    only. A hand-written whitelist would drift the day ``Client`` grows a
    parameter, and the drift would surface as a TypeError at exit 1.
    """
    global _CLIENT_PARAMS  # pylint: disable=global-statement
    if _CLIENT_PARAMS is None:
        # pylint: disable=import-outside-toplevel
        import inspect

        from dashscope import Client

        params = inspect.signature(Client.__init__).parameters
        _CLIENT_PARAMS = frozenset(params) - {"self"}
    return _CLIENT_PARAMS


def build_client(overrides: Mapping[str, Any]) -> Any:
    """Construct a :class:`dashscope.Client` from resolved CLI settings.

    *overrides* carries only the settings the user actually specified;
    the SDK's own resolution chain (explicit > env > config file) fills
    the rest.

    A key ``Client(...)`` does not take is applied to
    ``dashscope._config.settings`` instead, which is the tier
    ``resolve_config`` reads just below an explicit argument -- so ``-c``
    still outranks the environment and the config file, and the setting
    reaches the connection pool rather than blowing up the constructor.
    """
    # pylint: disable=import-outside-toplevel
    from dashscope import Client, _config

    accepted = _client_params()
    kwargs: Dict[str, Any] = {}
    for key, value in dict(overrides).items():
        if key in accepted:
            kwargs[key] = value
        elif hasattr(_config.settings, key):
            # The setter validates, so a bad value raises here rather than
            # being written into the pool.
            setattr(_config.settings, key, value)
        else:
            raise usage_error(
                f"configuration key {key!r} is neither a Client argument "
                f"nor a runtime setting; it cannot be overridden with -c",
            )
    return Client(user_agent_suffix=USER_AGENT_SUFFIX, **kwargs)
