# -*- coding: utf-8 -*-
"""Optional dependency ``loongsuite-util-genai`` and safe handler entry point.

This module is the *single* place where we integrate with
``loongsuite-util-genai``. It provides a hardened handler wrapper so that
**observability failures never
impact business logic**.

Debug environment variables
---------------------------

These are **diagnostics only**; unset in production unless troubleshooting.

``AGENTIC_RL_DEBUG_SPAN_BINDING``
    Set to ``1`` / ``true`` / ``yes`` to log span parent/child binding details
    (handler ``_WrappedCM`` and ``tools.trace_tool`` / ``observe_tool`` paths).

``AGENTIC_RL_DEBUG_LLM_OUTPUT``
    Set to ``1`` / ``true`` / ``yes`` / ``y`` in ``llm_openai`` to log
    completion /
    ``output_messages`` shapes (see ``_DEBUG_LLM_OUTPUT`` there).

``AGENTIC_RL_DEBUG_LLM_TOOL_CALLS``
    Set to ``1`` / ``true`` / ``yes`` / ``y`` in ``messages`` to log
    per-message ``tool_calls`` shapes and per-item ``tool_calls_to_parts``
    mapping (branch, argument types/lengths) when building LLM
    ``input_messages`` — for diagnosing empty ``tool_call.arguments`` in
    traces.

Recommended check before merging changes under ``…/observability/genai/``::

    PYTHONPATH=. pytest tests/observability/genai/ -q

(Requires the SDK repo layout and ``tests/observability/genai/`` contract
tests.)
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set
from typing_extensions import Literal

try:
    from opentelemetry.util.genai.extended_handler import (
        get_extended_telemetry_handler,
    )
    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
    from opentelemetry.util.genai.types import (
        InputMessage,
        OutputMessage,
        Text,
        ToolCall,
    )

    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    get_extended_telemetry_handler = None  # type: ignore[assignment]
    ExecuteToolInvocation = None  # type: ignore[assignment]
    # type: ignore[assignment]
    InputMessage = OutputMessage = Text = ToolCall = None
    GENAI_AVAILABLE = False

_logger = logging.getLogger(__name__)

_DEBUG_BINDING = os.environ.get(
    "AGENTIC_RL_DEBUG_SPAN_BINDING",
    "",
).strip().lower() in (
    "1",
    "true",
    "yes",
)


def _truncate_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


@dataclass(frozen=True)
class _SerializeLimits:
    max_depth: int = 6
    max_items: int = 200
    max_str_len: int = 8192


def to_jsonable(
    obj: Any,
    *,
    limits: _SerializeLimits = _SerializeLimits(),
) -> Any:
    """Best-effort conversion to a JSON-serializable, size-bounded structure.

    This is intentionally conservative: it prefers safety (never raising) and
    bounded output over perfect fidelity.
    """

    seen: Set[int] = set()

    # pylint: disable=too-many-branches
    # Serialization tree walk: many early returns / type branches by design.
    def _inner(  # pylint: disable=too-many-return-statements,
        # too-many-branches
        x: Any,
        depth: int,
    ) -> Any:
        if x is None or isinstance(x, (bool, int, float)):
            return x
        if isinstance(x, str):
            return _truncate_str(x, limits.max_str_len)
        if isinstance(x, bytes):
            return {
                "__type__": "bytes",
                "len": len(x),
                "preview": _truncate_str(repr(x[:64]), limits.max_str_len),
            }
        if isinstance(x, Enum):
            return x.value

        oid = id(x)
        if oid in seen:
            return "<recursion>"
        seen.add(oid)

        if depth >= limits.max_depth:
            return _truncate_str(repr(x), limits.max_str_len)

        if hasattr(x, "model_dump"):
            try:
                return _inner(x.model_dump(mode="json"), depth + 1)
            except Exception:
                try:
                    return _inner(x.model_dump(), depth + 1)
                except Exception:
                    pass
        if hasattr(x, "dict") and callable(getattr(x, "dict", None)):
            try:
                return _inner(x.Dict(), depth + 1)  # type: ignore[call-arg]
            except Exception:
                pass
        if isinstance(x, Dict):
            out: Dict[str, Any] = {}
            for i, (k, v) in enumerate(x.items()):
                if i >= limits.max_items:
                    out["__truncated__"] = True
                    break
                out[str(k)] = _inner(v, depth + 1)
            return out
        if isinstance(x, (list, tuple, set)):
            arr = []
            for i, item in enumerate(x):
                if i >= limits.max_items:
                    arr.append("<truncated>")
                    break
                arr.append(_inner(item, depth + 1))
            return arr

        return {
            "__type__": type(x).__name__,
            "__repr__": _truncate_str(repr(x), limits.max_str_len),
        }

    try:
        converted = _inner(obj, 0)
        json.dumps(converted, ensure_ascii=False)
        return converted
    except Exception:
        return _truncate_str(repr(obj), limits.max_str_len)


class SafeSpanProxy:
    """Proxy an OTel span-like object and suppress any errors from span
    operations."""

    def __init__(self, span: Any):
        self._span = span

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        return getattr(self._span, name)

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        try:
            if self._span is not None:
                self._span.set_status(*args, **kwargs)
        except Exception:
            return

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            if self._span is not None:
                self._span.set_attribute(key, to_jsonable(value))
        except Exception:
            return

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        try:
            if self._span is not None:
                safe_attrs = {
                    str(k): to_jsonable(v)
                    for k, v in (attributes or {}).items()
                }
                self._span.set_attributes(safe_attrs)
        except Exception:
            return

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if self._span is not None:
                safe_attrs = (
                    None
                    if attributes is None
                    else {
                        str(k): to_jsonable(v) for k, v in attributes.items()
                    }
                )
                self._span.add_event(name, attributes=safe_attrs, **kwargs)
        except Exception:
            return

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if self._span is not None:
                safe_attrs = (
                    None
                    if attributes is None
                    else {
                        str(k): to_jsonable(v) for k, v in attributes.items()
                    }
                )
                self._span.record_exception(
                    exception,
                    attributes=safe_attrs,
                    **kwargs,
                )
        except Exception:
            return


class SafeInvocationProxy:
    """Proxy the GenAI invocation object to enforce JSON-safe payloads.

    Span failures at the invocation boundary are suppressed.
    """

    _SERIALIZE_FIELDS = {
        "tool_call_arguments",
        "tool_call_result",
        "stop_sequences",
    }

    def __init__(self, inv: Any):
        object.__setattr__(self, "_inv", inv)

    def __getattr__(self, name: str) -> Any:
        if name == "span":
            span = getattr(object.__getattribute__(self, "_inv"), "span", None)
            return SafeSpanProxy(span) if span is not None else None
        return getattr(object.__getattribute__(self, "_inv"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        inv = object.__getattribute__(self, "_inv")
        try:
            if name in ("input_messages", "output_messages"):
                # Important: loongsuite-util-genai expects typed message
                # objects (InputMessage/OutputMessage) so it can read fields
                # like OutputMessage.finish_reason during span finalization.
                # If we eagerly convert them to dicts, handler.__exit__ may
                # crash and corrupt context propagation.
                #
                # We therefore only apply to_jsonable when the payload is
                # clearly not a list of message-like objects.
                v = value
                if isinstance(v, list):
                    all_msg_like = True
                    for m in v:
                        if m is None:
                            continue
                        if not hasattr(m, "role") or not hasattr(m, "parts"):
                            all_msg_like = False
                            break
                    if all_msg_like:
                        setattr(inv, name, v)
                    else:
                        setattr(inv, name, to_jsonable(v))
                else:
                    setattr(inv, name, to_jsonable(v))
            elif name in self._SERIALIZE_FIELDS:
                setattr(inv, name, to_jsonable(value))
            else:
                setattr(inv, name, value)
        except Exception:
            # Observability should never break business logic
            return


class _SimpleCircuitBreaker:
    """Lightweight, process-local breaker to avoid repeated observability
    failures."""

    def __init__(self) -> None:
        self._fail_count = 0
        self._disabled_until = 0.0

    def allow(self) -> bool:
        return time.time() >= self._disabled_until

    def record_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count >= 5:
            self._disabled_until = time.time() + 60.0
            self._fail_count = 0


_breaker = _SimpleCircuitBreaker()


class SafeHandlerProxy:
    """Proxy the LoongSuite GenAI handler to guarantee silent degradation."""

    def __init__(self, handler: Any):
        self._handler = handler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handler, name)

    def llm(self, *args: Any, **kwargs: Any) -> Any:
        if not _breaker.allow():
            return _noop_cm()
        try:
            cm = self._handler.llm(*args, **kwargs)
            return _wrap_cm(cm, label="llm")
        except Exception as e:
            _breaker.record_failure()
            _logger.debug("[OTel] llm init error: %s", e)
            return _noop_cm()

    def execute_tool(self, *args: Any, **kwargs: Any) -> Any:
        if not _breaker.allow():
            return _noop_cm()
        try:
            cm = self._handler.execute_tool(*args, **kwargs)
            return _wrap_cm(cm, label="execute_tool")
        except Exception as e:
            _breaker.record_failure()
            _logger.debug("[OTel] execute_tool init error: %s", e)
            return _noop_cm()


class NoopHandler:
    """A drop-in handler that does nothing and never raises."""

    def llm(self, *_: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        return _noop_cm()

    def execute_tool(
        self,
        *_: Any,
        **_kwargs: Any,
    ) -> Any:  # pragma: no cover
        return _noop_cm()


class _NoopCM:
    def __enter__(self) -> Any:
        return SafeInvocationProxy(type("NoopInv", (), {})())

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False


def _noop_cm() -> _NoopCM:
    return _NoopCM()


class _WrappedCM:
    def __init__(self, cm: Any, label: str):
        self._cm = cm
        self._label = label
        self._last_inv: Any = None

    def __enter__(self) -> Any:
        try:
            inv = self._cm.__enter__()
            self._last_inv = inv
            return SafeInvocationProxy(inv)
        except Exception as e:
            _breaker.record_failure()
            if _DEBUG_BINDING:
                _logger.warning(
                    "[OTel] %s __enter__ error (degrading to noop): %s",
                    self._label,
                    e,
                )
            else:
                _logger.debug("[OTel] %s __enter__ error: %s", self._label, e)
            self._last_inv = None
            return SafeInvocationProxy(type("NoopInv", (), {})())

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        # Delegate: business exceptions pass through; __exit__ errors
        # suppressed below.
        try:
            return bool(self._cm.__exit__(exc_type, exc, tb))
        except Exception as e:
            _breaker.record_failure()
            if _DEBUG_BINDING:
                _logger.warning(
                    "[OTel] %s __exit__ error (suppressed): %s",
                    self._label,
                    e,
                )
            else:
                _logger.debug("[OTel] %s __exit__ error: %s", self._label, e)
            # Critical: if the underlying handler failed in __exit__,
            # it may have skipped context detach / span end. That would
            # corrupt downstream span parenting (e.g. tools appearing as
            # siblings) and may prevent spans from being exported.
            # Best-effort cleanup here.
            try:
                inv = self._last_inv
                token = (
                    getattr(inv, "context_token", None)
                    if inv is not None
                    else None
                )
                span = getattr(inv, "span", None) if inv is not None else None
                if token is not None:
                    try:
                        from opentelemetry import context as otel_context

                        otel_context.detach(token)
                    except Exception:
                        pass
                if span is not None:
                    try:
                        span.end()
                    except Exception:
                        pass
            except Exception:
                pass
            return False


def _wrap_cm(cm: Any, *, label: str) -> Any:
    # We need a proxy on __enter__ return value, not only a generic safe_cm.
    return _WrappedCM(cm, label=label)


def get_handler() -> Any:
    """Return a hardened GenAI telemetry handler.

    The returned handler is always safe to use from business code: any internal
    failures will degrade to no-op rather than propagating exceptions.
    """
    if not GENAI_AVAILABLE:
        return NoopHandler()
    try:
        raw = get_extended_telemetry_handler()
        if raw is None:
            return NoopHandler()
        return SafeHandlerProxy(raw)
    except Exception as e:
        _logger.debug("[OTel] get_handler failed: %s", e)
        return NoopHandler()


__all__ = [
    "GENAI_AVAILABLE",
    "ExecuteToolInvocation",
    "InputMessage",
    "OutputMessage",
    "Text",
    "ToolCall",
    "NoopHandler",
    "SafeHandlerProxy",
    "SafeInvocationProxy",
    "SafeSpanProxy",
    "get_handler",
    "to_jsonable",
]
