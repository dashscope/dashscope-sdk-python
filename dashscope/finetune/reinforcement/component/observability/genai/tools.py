# -*- coding: utf-8 -*-
"""Tool / function-call ``execute_tool`` span decorator and ``trace_tool``
patcher.

Two complementary APIs:

- ``@observe_tool``: Decorator for user-defined tool functions.
- ``trace_tool``: Monkey-patch LangChain BaseTool objects for automatic
  tracing.

They are orthogonal and can be used together (though typically only one is
needed).

Phase-A diagnostics (nested ``execute_tool`` investigation): set environment
variable ``AGENTIC_RL_DEBUG_TRACE_TOOL=true`` to log ``wrap_ainvoke`` /
``wrap_invoke`` lines; ``AGENTIC_RL_DEBUG_TRACE_TOOL_FALLBACK=true`` adds or
selects ``fallback_span`` lines. ``AGENTIC_RL_DEBUG_SPAN_BINDING`` uses the
same truthy values as other observability env flags (``true`` / ``1`` /
``yes`` / ``y`` / ``on``).

Deduplication (default **on**): LangChain often calls ``ainvoke`` then
``invoke`` for sync tools, which would create two ``execute_tool`` spans. The
SDK suppresses the inner duplicate by default. Set
``AGENTIC_RL_TRACE_TOOL_NO_DEDUP=true`` only to disable this (e.g. debugging).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import os
import threading
from contextlib import ExitStack, nullcontext
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union

from typing_extensions import Literal

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import use_span

    _OTEL_USE_SPAN = True
except ImportError:  # pragma: no cover
    otel_trace = None  # type: ignore[assignment]
    use_span = None  # type: ignore[assignment]
    _OTEL_USE_SPAN = False

from opentelemetry.trace.status import Status, StatusCode

from dashscope.finetune.reinforcement.component.observability import (
    genai,
    tracing,
)

# pylint: disable=protected-access
_core = genai._core

F = TypeVar("F", bound=Callable[..., Any])

# Runnable tool call first argument type (name ``input`` is required by LC).
_In = Union[str, Dict[str, Any]]

# Module-level logger for trace_tool warnings
_logger = logging.getLogger(__name__)

# Truthy env values for observability toggles (aligned with
# ``messages._env_truthy``).
_ENV_TRUTHY_VALUES = ("true", "1", "yes", "y", "on")

_DEBUG_BINDING = (
    os.environ.get("AGENTIC_RL_DEBUG_SPAN_BINDING", "").strip().lower()
    in _ENV_TRUTHY_VALUES
)


# Phase-A diagnostics: nested ``execute_tool`` spans (ainvoke vs invoke vs
# fallback). Set ``AGENTIC_RL_DEBUG_TRACE_TOOL=true`` for wrap_ainvoke /
# wrap_invoke lines. Set ``AGENTIC_RL_DEBUG_TRACE_TOOL_FALLBACK=true`` for
# fallback_span only (or together with above).
def _env_flag(name: str) -> bool:
    """Truth-y values aligned with ``messages._env_truthy`` / other
    observability env parsers."""
    return os.environ.get(name, "").strip().lower() in _ENV_TRUTHY_VALUES


_DEBUG_TRACE_TOOL_WRAP = _env_flag("AGENTIC_RL_DEBUG_TRACE_TOOL")
_DEBUG_TRACE_TOOL_FALLBACK = (
    _env_flag("AGENTIC_RL_DEBUG_TRACE_TOOL_FALLBACK") or _DEBUG_TRACE_TOOL_WRAP
)

# Default: dedupe nested ainvoke→invoke double spans. Escape hatch for rare
# debugging only.
_TRACE_TOOL_NO_DEDUP = _env_flag("AGENTIC_RL_TRACE_TOOL_NO_DEDUP")

# Same-thread nesting (``invoke`` called synchronously under ``ainvoke``
# without a thread hop).
_TRACE_TOOL_CTX_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "agentic_rl_trace_tool_ctx_depth",
    default=0,
)

# Cross-thread / LangGraph: while ``wrapped_ainvoke`` awaits
# ``orig_ainvoke``, refcount per OTel trace id so ``wrapped_invoke`` in a
# worker thread can detect an outer async tool layer.
_outer_async_tool_lock = threading.Lock()
_outer_async_tool_depth: Dict[str, int] = {}

_trace_tool_seq_lock = threading.Lock()
_trace_tool_seq = 0


def _next_trace_tool_seq() -> int:
    global _trace_tool_seq
    with _trace_tool_seq_lock:
        _trace_tool_seq += 1
        return _trace_tool_seq


def _current_otel_trace_id_hex() -> str:
    if otel_trace is None:
        return "-"
    try:
        from opentelemetry.trace import format_trace_id

        sp = otel_trace.get_current_span()
        ctx = sp.get_span_context()
        if ctx.is_valid:
            return format_trace_id(ctx.trace_id)
    except Exception:
        pass
    return "-"


def _trace_hex_or_none() -> Optional[str]:
    h = _current_otel_trace_id_hex()
    return None if h in ("-", "") else h


def _incr_outer_async_layer(trace_key: str) -> None:
    with _outer_async_tool_lock:
        _outer_async_tool_depth[trace_key] = (
            _outer_async_tool_depth.get(trace_key, 0) + 1
        )


def _decr_outer_async_layer(trace_key: str) -> None:
    with _outer_async_tool_lock:
        c = _outer_async_tool_depth.get(trace_key, 0) - 1
        if c <= 0:
            _outer_async_tool_depth.pop(trace_key, None)
        else:
            _outer_async_tool_depth[trace_key] = c


def _outer_async_layer_depth(trace_key: str) -> int:
    with _outer_async_tool_lock:
        return _outer_async_tool_depth.get(trace_key, 0)


def _otel_current_span_looks_like_execute_tool_outer() -> bool:
    """True when OTel current span is likely our outer ``execute_tool``.

    Uses a name substring match (``execute_tool``). Exporter or SDK span
    naming changes, or unrelated spans whose names contain the same substring,
    could false-positive; this is a last-resort hint and is combined with
    other dedup signals in ``_should_skip_inner_tool_span``.
    """
    if otel_trace is None:
        return False
    try:
        sp = otel_trace.get_current_span()
        if sp is None:
            return False
        if not sp.is_recording():
            return False
        name = (getattr(sp, "name", None) or "").lower()
        return "execute_tool" in name
    except Exception:
        return False


def _should_skip_inner_tool_span(_tool_name: str, _config: Any) -> bool:
    """Suppress duplicate ``execute_tool`` for LangChain ``ainvoke`` →
    ``invoke``.

    When LangChain has already wrapped the async path, inner sync invokes are
    skipped.
    """
    if _TRACE_TOOL_NO_DEDUP:
        return False
    if _TRACE_TOOL_CTX_DEPTH.get() > 0:
        return True
    tk = _trace_hex_or_none()
    if tk is not None and _outer_async_layer_depth(tk) > 0:
        return True
    if _otel_current_span_looks_like_execute_tool_outer():
        return True
    return False


def _extract_rollout_id_from_config(config: Any) -> str:
    """Best-effort ``rollout_id`` from LangChain ``RunnableConfig`` or
    dict-like config."""
    if config is None:
        return "-"
    try:
        if isinstance(config, Dict):
            md = config.get("metadata") or {}
            if isinstance(md, Dict) and md.get("rollout_id") is not None:
                return str(md["rollout_id"])
            cfg = config.get("configurable")
            if isinstance(cfg, Dict):
                meta = cfg.get("metadata")
                if (
                    isinstance(meta, Dict)
                    and meta.get("rollout_id") is not None
                ):
                    return str(meta["rollout_id"])
        else:
            md = getattr(config, "metadata", None) or {}
            if isinstance(md, Dict) and md.get("rollout_id") is not None:
                return str(md["rollout_id"])
            cfg = getattr(config, "configurable", None)
            if isinstance(cfg, Dict):
                meta = cfg.get("metadata")
                if (
                    isinstance(meta, Dict)
                    and meta.get("rollout_id") is not None
                ):
                    return str(meta["rollout_id"])
    except Exception:
        pass
    return "-"


def _emit_trace_tool_debug_line(
    phase: str,
    tool_name: str,
    config: Any,
    *,
    extra: Optional[str] = None,
) -> None:
    """Single-line log for FC grep: ``[AGENTIC_RL_TRACE_TOOL]``."""
    seq = _next_trace_tool_seq()
    tid = threading.get_ident()
    trace_id = _current_otel_trace_id_hex()
    rid = _extract_rollout_id_from_config(config)
    suffix = f" {extra}" if extra else ""
    _logger.info(
        (
            "[AGENTIC_RL_TRACE_TOOL] phase=%s tool=%s seq=%s tid=%s "
            "trace_id=%s rollout_id=%s%s"
        ),
        phase,
        tool_name,
        seq,
        tid,
        trace_id,
        rid,
        suffix,
    )


def _otel_span_id_hex(span: Any) -> Optional[str]:
    """Hex span id for an OTel span-like object, or None."""
    if span is None:
        return None
    try:
        from opentelemetry.trace import format_span_id

        ctx = span.get_span_context()
        if ctx.is_valid:
            return format_span_id(ctx.span_id)
    except Exception:
        return None
    return None


def _current_span_id_hex() -> Optional[str]:
    if not _DEBUG_BINDING or otel_trace is None:
        return None
    try:
        return _otel_span_id_hex(otel_trace.get_current_span())
    except Exception:
        return None


def _invocation_span_id_hex(invocation: Any) -> Optional[str]:
    if not _DEBUG_BINDING:
        return None
    return _otel_span_id_hex(getattr(invocation, "span", None))


def _debug_binding_point(label: str, invocation: Any) -> None:
    if not _DEBUG_BINDING:
        return
    try:
        cur = _current_span_id_hex()
        inv = _invocation_span_id_hex(invocation)
        span = getattr(invocation, "span", None)
        rec = None
        try:
            rec = span.is_recording() if span is not None else None
        except Exception:
            rec = None
        _logger.info(
            "[debug_span_binding] %s current_span_id=%s invocation_span_id=%s "
            "invocation_span_type=%s is_recording=%s",
            label,
            cur,
            inv,
            type(span).__name__ if span is not None else None,
            rec,
        )
    except Exception:
        return


class _NoopCM:
    """Local no-op CM for fallback span binding."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> Literal[False]:
        return False


class _FallbackSpanCM:
    """Context manager that yields an OTel span (or None) for fallback tool
    spans."""

    def __init__(self, cm: Any):
        self._cm = cm
        self.span: Any = None

    def __enter__(self) -> Any:
        try:
            self.span = self._cm.__enter__()
            return self.span
        except Exception:
            self.span = None
            return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            return bool(self._cm.__exit__(exc_type, exc, tb))
        except Exception:
            return False


def _maybe_start_fallback_tool_span(tool_name: str) -> Any:
    """Start an OTel execute_tool span when the GenAI handler fails to provide
    one.

    This preserves silent degradation (never raises) and restores UI visibility
    of tool spans when loongsuite-util-genai handler is degraded / no-op.
    """
    try:
        tracer = tracing.get_tracer()
        if tracer is None:
            return _NoopCM()
        return _FallbackSpanCM(
            tracer.start_as_current_span(f"execute_tool {tool_name}"),
        )
    except Exception:
        return _NoopCM()


def _best_effort_set_ok(span: Any) -> None:
    if span is None:
        return
    try:
        span.set_status(Status(StatusCode.OK))
    except Exception:
        return


def _best_effort_set_error(span: Any, err: BaseException) -> None:
    if span is None:
        return
    try:
        span.set_status(Status(StatusCode.ERROR, str(err)))
    except Exception:
        pass
    try:
        span.record_exception(err)
    except Exception:
        return


def _best_effort_set_attribute(span: Any, key: str, value: Any) -> None:
    if span is None:
        return
    try:
        span.set_attribute(key, _core.to_jsonable(value))
    except Exception:
        return


def _unwrap_span(span: Any) -> Any:
    """Return the underlying OTel span when wrapped by safety proxies."""
    try:
        raw = getattr(span, "_span", None)
        return raw if raw is not None else span
    except Exception:
        return span


def _best_effort_set_tool_json(span: Any, key: str, value: Any) -> None:
    """Tool attributes are required but must be primitive types for OTel.

    We encode structured values as a compact JSON string.
    """
    if span is None:
        return
    try:
        payload = _core.to_jsonable(value)
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        span.set_attribute(key, encoded)
    except Exception:
        return


class _ToolSpanScope:
    """Unified tool-span scope.

    - Prefer the handler-provided invocation span when available.
    - Otherwise, create a fallback OTel span so tools remain visible in UI.
    - Always best-effort set OK/ERROR on whichever span exists.
    """

    def __init__(self, invocation: Any, tool_name: str):
        self._invocation = invocation
        self._tool_name = tool_name
        self._stack = ExitStack()
        self.invocation_span: Any = None
        self.fallback_span: Any = None

    def __enter__(self) -> "_ToolSpanScope":
        self.invocation_span = getattr(self._invocation, "span", None)
        raw_span = _unwrap_span(self.invocation_span)
        # 1) Bind handler span as current (if provided).
        cm_bind = (
            use_span(raw_span, end_on_exit=False)
            if (_OTEL_USE_SPAN and raw_span is not None)
            else nullcontext()
        )
        self._stack.enter_context(cm_bind)
        # 2) If handler degraded to no-span, start a fallback span.
        if self.invocation_span is None:
            if _DEBUG_TRACE_TOOL_FALLBACK:
                _emit_trace_tool_debug_line(
                    "fallback_span",
                    self._tool_name,
                    None,
                    extra="invocation_span=null",
                )
            fb_cm = _maybe_start_fallback_tool_span(self._tool_name)
            try:
                self.fallback_span = self._stack.enter_context(fb_cm)
            except Exception:
                self.fallback_span = None

        # Mini spec required fields for TOOL spans:
        # - gen_ai.tool.name (Y)
        # - gen_ai.tool.call.arguments (Y)
        # The result is written on __exit__ (when available).
        effective = self.invocation_span or self.fallback_span
        _best_effort_set_attribute(
            effective,
            "gen_ai.tool.name",
            self._tool_name,
        )
        try:
            args = getattr(self._invocation, "tool_call_arguments", None)
            if args is not None:
                _best_effort_set_tool_json(
                    effective,
                    "gen_ai.tool.call.arguments",
                    args,
                )
        except Exception:
            pass
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        # Fill TOOL required result field best-effort.
        effective = self.invocation_span or self.fallback_span
        try:
            res = getattr(self._invocation, "tool_call_result", None)
            if res is not None:
                _best_effort_set_tool_json(
                    effective,
                    "gen_ai.tool.call.result",
                    res,
                )
        except Exception:
            pass
        # Mark status on both spans best-effort.
        if exc is None:
            _best_effort_set_ok(self.invocation_span)
            _best_effort_set_ok(self.fallback_span)
        else:
            _best_effort_set_error(self.invocation_span, exc)
            _best_effort_set_error(self.fallback_span, exc)
        try:
            return bool(self._stack.__exit__(exc_type, exc, tb))
        except Exception:
            return False


def _bind_tool_arguments(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: Dict,
) -> Dict:
    """Bind positional/keyword arguments to ``fn``'s signature and return a
    JSON-serialisable dict.

    ``self`` / ``cls`` are stripped from the result.  Falls back to an error
    sentinel on any binding failure so that the span is still emitted with a
    best-effort payload.
    """
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        param_names = list(sig.parameters.keys())
        raw = Dict(bound.arguments)
        if param_names and param_names[0] in ("self", "cls"):
            raw = {k: v for k, v in raw.items() if k not in ("self", "cls")}
        return _json_serializable(raw)
    except Exception:
        return {"_error": "could_not_bind_arguments"}


def _json_serializable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="json")
            except Exception:
                pass
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def observe_tool(
    _fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    handler: Any = None,
) -> Any:
    """
    Decorator that wraps a tool / function-call implementation in an
    ``execute_tool`` GenAI span.

    - ``name``    : span tool name; defaults to ``fn.__name__``.
    - ``provider``: optional provider label written to the span (e.g.
      ``"my-plugin"``).
    - ``handler`` : override the default LoongSuite telemetry handler.

    Supports both sync and async functions.  All positional / keyword
    arguments are bound to the function signature, serialised to JSON, and
    recorded as ``tool_call_arguments`` on the span; ``self`` / ``cls`` are
    excluded automatically. No-op when ``ENABLE_TRAJECTORY`` is unset or
    ``loongsuite-util-genai`` is not installed.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not tracing.is_tracing_enabled() or not _core.GENAI_AVAILABLE:
                return fn(*args, **kwargs)

            h = handler if handler is not None else _core.get_handler()
            tool_name = name or fn.__name__
            arguments = _bind_tool_arguments(fn, args, kwargs)

            inv = _core.ExecuteToolInvocation(
                tool_name=tool_name,
                tool_call_arguments=arguments,
            )
            if provider:
                inv.provider = provider

            with h.execute_tool(inv) as invocation:
                with _ToolSpanScope(invocation, tool_name) as _:
                    _debug_binding_point(
                        "observe_tool_sync:entered",
                        invocation,
                    )
                    _debug_binding_point(
                        "observe_tool_sync:after_use_span",
                        invocation,
                    )
                    tracing.log_trace_id(f"execute_tool:{tool_name}")
                    result = fn(*args, **kwargs)
                    invocation.tool_call_result = _json_serializable(result)
                    # Prefer handler span status if present; fallback status
                    # handled by scope.
                    if (
                        hasattr(invocation, "span")
                        and invocation.span is not None
                    ):
                        invocation.span.set_status(Status(StatusCode.OK))
                    return result

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not tracing.is_tracing_enabled() or not _core.GENAI_AVAILABLE:
                return await fn(*args, **kwargs)

            h = handler if handler is not None else _core.get_handler()
            tool_name = name or fn.__name__
            arguments = _bind_tool_arguments(fn, args, kwargs)

            inv = _core.ExecuteToolInvocation(
                tool_name=tool_name,
                tool_call_arguments=arguments,
            )
            if provider:
                inv.provider = provider

            with h.execute_tool(inv) as invocation:
                with _ToolSpanScope(invocation, tool_name) as _:
                    _debug_binding_point(
                        "observe_tool_async:entered",
                        invocation,
                    )
                    _debug_binding_point(
                        "observe_tool_async:after_use_span",
                        invocation,
                    )
                    tracing.log_trace_id(f"execute_tool:{tool_name}")
                    result = await fn(*args, **kwargs)
                    invocation.tool_call_result = _json_serializable(result)
                    if (
                        hasattr(invocation, "span")
                        and invocation.span is not None
                    ):
                        invocation.span.set_status(Status(StatusCode.OK))
                    return result

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


# ============================================================================
# trace_tool: Monkey-patch LangChain BaseTool objects
# ============================================================================

_PATCHED_ATTR = "_agentic_rl_genai_tool_patched"

# Track unsupported types we've already warned about (avoid log spam)
_warned_unsupported_types: Set[type] = set()


def trace_tool(
    tools: Any,
    *,
    provider: Optional[str] = None,
) -> None:
    """
    Enable tracing on LangChain BaseTool objects.

    Automatically detects and patches the following shapes:

    - ``list[BaseTool]`` / ``tuple[BaseTool]`` — each tool is patched
    - ``ToolNode`` (LangGraph) — internal tools_by_name is expanded and patched
    - ``Dict[str, BaseTool]`` — each value is patched
    - Single ``BaseTool`` instance — patched directly

    The function is idempotent — patching the same object twice is safe.
    No-op when tracing is disabled or ``loongsuite-util-genai`` is not
    installed.

    MCP tools (from ``langchain-mcp-adapters``) are automatically detected
    and the provider is set to ``"mcp"`` unless overridden.

    Args:
        tools: A tool object, list of tools, ToolNode, or dict of tools.
        provider: Optional provider label written to spans (e.g. ``"mcp"``,
            ``"custom"``). If ``None`` and the tool is detected as MCP,
            ``"mcp"`` is used automatically.

    Example:
        >>> from langchain_mcp_adapters.client import MultiServerMCPClient
        >>> client = MultiServerMCPClient(servers)
        >>> tools = await client.get_tools()
        >>> trace_tool(tools)  # One-liner for MCP tools
        >>>
        >>> # Or with ToolNode
        >>> from langgraph.prebuilt import ToolNode
        >>> trace_tool(ToolNode(tools))
        >>>
        >>> # Custom provider
        >>> trace_tool(my_custom_tool, provider="my-plugin")

    Note:
        This patches ``BaseTool.invoke`` and ``BaseTool.ainvoke`` methods.
        For non-LangChain tools, use the ``@observe_tool`` decorator instead.
    """
    if not tracing.is_tracing_enabled() or not _core.GENAI_AVAILABLE:
        return

    # 1. ToolNode (LangGraph) — has .tools_by_name dict
    if hasattr(tools, "tools_by_name") and isinstance(
        getattr(tools, "tools_by_name", None),
        Dict,
    ):
        for tool in tools.tools_by_name.values():
            _patch_single_tool(tool, provider)
        return

    # 2. list / tuple
    if isinstance(tools, (list, tuple)):
        for tool in tools:
            _patch_single_tool(tool, provider)
        return

    # 3. dict (name → tool mapping)
    if isinstance(tools, Dict):
        for tool in tools.values():
            _patch_single_tool(tool, provider)
        return

    # 4. Single tool
    _patch_single_tool(tools, provider)


def _is_base_tool(obj: Any) -> bool:
    """Duck-typing check for LangChain BaseTool.

    A LangChain BaseTool must have:
    - ``name`` attribute (tool identifier)
    - ``invoke`` or ``ainvoke`` callable method (Runnable interface)

    This duck-typing approach is framework-agnostic and works across
    LangChain versions without import-time dependencies.
    """
    if not hasattr(obj, "name"):
        return False
    if not callable(getattr(obj, "invoke", None)):
        if not callable(getattr(obj, "ainvoke", None)):
            return False
    return True


def _detect_mcp_tool(tool: Any) -> bool:
    """Detect if a tool originates from MCP (langchain-mcp-adapters).

    MCP tools have specific markers:
    - Internal ``_mcp_tool`` attribute (langchain-mcp-adapters implementation
      detail)
    - Description containing MCP-related keywords

    This detection is best-effort and may need updates as
    langchain-mcp-adapters evolves.
    """
    # Direct marker from langchain-mcp-adapters
    if hasattr(tool, "_mcp_tool"):
        return True

    # Heuristic: check description for MCP keywords
    desc = getattr(tool, "description", "") or ""
    if isinstance(desc, str) and "mcp" in desc.lower():
        return True

    return False


def _patch_single_tool(tool: Any, provider: Optional[str]) -> None:
    """Patch a single tool object's invoke/ainvoke methods.

    This is the core implementation for ``trace_tool``. It:
    1. Validates the tool is a LangChain BaseTool (duck-typing)
    2. Checks idempotency guard
    3. Auto-detects MCP tools and sets provider
    4. Patches invoke (sync) and ainvoke (async) methods

    Uses ``object.__setattr__`` to bypass pydantic's attribute interception,
    and wraps all operations in try-except for silent degradation.
    """
    try:
        if not _is_base_tool(tool):
            _log_unsupported_tool(tool)
            return

        # Idempotency guard — skip if already patched
        if getattr(tool, _PATCHED_ATTR, False):
            return

        # Auto-detect MCP tools
        effective_provider = provider
        if effective_provider is None and _detect_mcp_tool(tool):
            effective_provider = "mcp"

        tool_name_for_log = getattr(tool, "name", str(tool))

        # W0622: keep the redefined-builtin pragma on the same physical line as
        # the ``input`` parameter; if ``Union[...]`` wraps so ``input`` sits
        # alone on a line, a trailing pragma is ignored.

        # Patch ainvoke (async) — this is the primary entry point for LangGraph
        # Use object.__setattr__ to bypass pydantic's attribute interception
        if callable(getattr(tool, "ainvoke", None)):
            orig_ainvoke = tool.ainvoke

            # Runnable signature uses ``input`` as the canonical argument name.
            # fmt: off
            @functools.wraps(orig_ainvoke)
            async def wrapped_ainvoke(
                input: _In,  # pylint: disable=redefined-builtin
                config: Any = None,
                **kwargs: Any,
            ) -> Any:
                return await _run_tool_with_span_async(
                    tool=tool,
                    orig_fn=orig_ainvoke,
                    tool_input=input,
                    config=config,
                    provider=effective_provider,
                    **kwargs,
                )

            # fmt: on
            object.__setattr__(tool, "ainvoke", wrapped_ainvoke)

        # Patch invoke (sync) — for synchronous usage
        if callable(getattr(tool, "invoke", None)):
            orig_invoke = tool.invoke

            # Runnable signature uses ``input`` as the canonical argument name.
            # fmt: off
            @functools.wraps(orig_invoke)
            def wrapped_invoke(
                input: _In,  # pylint: disable=redefined-builtin
                config: Any = None,
                **kwargs: Any,
            ) -> Any:
                return _run_tool_with_span_sync(
                    tool=tool,
                    orig_fn=orig_invoke,
                    tool_input=input,
                    config=config,
                    provider=effective_provider,
                    **kwargs,
                )

            # fmt: on
            object.__setattr__(tool, "invoke", wrapped_invoke)

        # Mark as patched
        object.__setattr__(tool, _PATCHED_ATTR, True)

        _logger.debug(
            "trace_tool: patched %r (provider=%s)",
            tool_name_for_log,
            effective_provider or "none",
        )
    except Exception as e:
        # Silent degradation: observability should never break business logic
        tool_name = getattr(tool, "name", str(tool))
        _logger.warning(
            "trace_tool: failed to patch tool %r: %s. Skipping tracing.",
            tool_name,
            e,
        )


def _run_tool_with_span_sync(
    tool: Any,
    orig_fn: Callable[..., Any],
    tool_input: Union[str, Dict[str, Any]],
    config: Any,
    provider: Optional[str],
    **kwargs: Any,
) -> Any:
    """Execute a tool synchronously within an execute_tool GenAI span."""
    tool_name = getattr(tool, "name", str(tool))
    if _DEBUG_TRACE_TOOL_WRAP:
        _emit_trace_tool_debug_line("wrap_invoke", tool_name, config)

    if not tracing.is_tracing_enabled() or not _core.GENAI_AVAILABLE:
        return orig_fn(tool_input, config=config, **kwargs)

    if _should_skip_inner_tool_span(tool_name, config):
        if _DEBUG_TRACE_TOOL_WRAP:
            _emit_trace_tool_debug_line(
                "wrap_invoke_skipped",
                tool_name,
                config,
                extra="dedup",
            )
        return orig_fn(tool_input, config=config, **kwargs)

    h = _core.get_handler()
    arguments = _extract_tool_arguments(tool_input)

    inv = _core.ExecuteToolInvocation(
        tool_name=tool_name,
        tool_call_arguments=arguments,
    )
    if provider:
        inv.provider = provider

    with h.execute_tool(inv) as invocation:
        with _ToolSpanScope(invocation, tool_name) as _:
            _debug_binding_point("trace_tool_sync:entered", invocation)
            _debug_binding_point("trace_tool_sync:after_use_span", invocation)
            tracing.log_trace_id(f"execute_tool:{tool_name}")
            result = orig_fn(tool_input, config=config, **kwargs)
            invocation.tool_call_result = _json_serializable(result)
            if hasattr(invocation, "span") and invocation.span is not None:
                invocation.span.set_status(Status(StatusCode.OK))
            return result


async def _run_tool_with_span_async(
    tool: Any,
    orig_fn: Callable[..., Any],
    tool_input: Union[str, Dict[str, Any]],
    config: Any,
    provider: Optional[str],
    **kwargs: Any,
) -> Any:
    """Execute a tool asynchronously within an execute_tool GenAI span."""
    tool_name = getattr(tool, "name", str(tool))
    if _DEBUG_TRACE_TOOL_WRAP:
        _emit_trace_tool_debug_line("wrap_ainvoke", tool_name, config)

    if not tracing.is_tracing_enabled() or not _core.GENAI_AVAILABLE:
        # type: ignore[misc]
        return await orig_fn(tool_input, config=config, **kwargs)

    trace_key = _trace_hex_or_none()
    ctx_token: Optional[contextvars.Token[int]] = None
    if not _TRACE_TOOL_NO_DEDUP:
        if trace_key is not None:
            _incr_outer_async_layer(trace_key)
        ctx_token = _TRACE_TOOL_CTX_DEPTH.set(_TRACE_TOOL_CTX_DEPTH.get() + 1)
    try:
        h = _core.get_handler()
        arguments = _extract_tool_arguments(tool_input)

        inv = _core.ExecuteToolInvocation(
            tool_name=tool_name,
            tool_call_arguments=arguments,
        )
        if provider:
            inv.provider = provider

        with h.execute_tool(inv) as invocation:
            with _ToolSpanScope(invocation, tool_name) as _:
                _debug_binding_point("trace_tool_async:entered", invocation)
                _debug_binding_point(
                    "trace_tool_async:after_use_span",
                    invocation,
                )
                tracing.log_trace_id(f"execute_tool:{tool_name}")
                # type: ignore[misc]
                result = await orig_fn(tool_input, config=config, **kwargs)
                invocation.tool_call_result = _json_serializable(result)
                if hasattr(invocation, "span") and invocation.span is not None:
                    invocation.span.set_status(Status(StatusCode.OK))
                return result
    finally:
        if not _TRACE_TOOL_NO_DEDUP:
            if ctx_token is not None:
                _TRACE_TOOL_CTX_DEPTH.reset(ctx_token)
            if trace_key is not None:
                _decr_outer_async_layer(trace_key)


def _extract_tool_arguments(
    tool_input: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract tool arguments from input (str or dict).

    LangChain BaseTool.invoke accepts either:
    - A string (single-argument tool)
    - A dict of named arguments
    """
    if isinstance(tool_input, str):
        return {"input": tool_input}
    if isinstance(tool_input, Dict):
        # Ensure the payload is JSON-serializable for the GenAI handler.
        return _core.to_jsonable(tool_input)
    # Fallback: try to serialize
    return {"_raw": _json_serializable(tool_input)}


def _log_unsupported_tool(tool: Any) -> None:
    """Log a warning for unsupported tool types (once per type)."""
    tool_type = type(tool)
    if tool_type in _warned_unsupported_types:
        return

    _warned_unsupported_types.add(tool_type)
    _logger.warning(
        "trace_tool: skipping unsupported object %r. "
        "Expected LangChain BaseTool with .name and .invoke/.ainvoke. "
        "Use @observe_tool decorator for custom tool functions.",
        tool_type,
    )


__all__ = ["observe_tool", "trace_tool"]
