# -*- coding: utf-8 -*-
"""
OpenTelemetry tracing for rollout / reward processors.

OpenTelemetry ships with the SDK; enable tracing with:

- ``ENABLE_TRAJECTORY=true``

Optional: ``AGENTIC_RL_LOG_TRACE_ID=true`` enables ``logger.info`` lines that
print the current W3C ``trace_id`` on processor / GenAI spans (see
:func:`log_trace_id`). Default is off for stable production workloads.

With the SDK installed, :func:`ensure_agentic_rl_baggage_span_processor`
registers :class:`~opentelemetry.processor.baggage.BaggageSpanProcessor` so
:func:`rollout_context` can stamp
``bailian.agentic_rl.rollout_id`` / ``sample_id`` / ``attempt_id`` onto all
descendant spans as both W3C Baggage entries and span attributes.

Span naming conventions (aligned with Bailian Agentic RL Tracing
specification):
- Entry root span  : ``invoke_agentic_rl``  (``gen_ai.span.kind=CHAIN``)
- Rollout processor: ``invoke_rollout <ClassName>``
  (``gen_ai.span.kind=ROLLOUT``)
- Reward processor : ``invoke_reward <ClassName>``
  (``gen_ai.span.kind=REWARD``)
"""

from __future__ import annotations

import json
import os
import threading
import contextvars
import asyncio
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Iterator, Optional, Tuple

from ...common.log import logger
from ...common.model_types import (
    FunctionType as FuncType,
)
from ..data.base_data_model import (
    BaseDataModel,
)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind as OtelSpanKind
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    otel_trace = None  # type: ignore[misc, assignment]
    OtelSpanKind = None  # type: ignore[misc, assignment]
    Status = None  # type: ignore[misc, assignment]
    StatusCode = None  # type: ignore[misc, assignment]
    _OTEL_AVAILABLE = False

_ENV_ENABLE_TRAJECTORY_SHORT = "ENABLE_TRAJECTORY"
_ENV_FORCE_FLUSH_MODE = "AGENTIC_RL_FORCE_FLUSH_MODE"
_ENV_FORCE_FLUSH_TIMEOUT_MS = "AGENTIC_RL_FORCE_FLUSH_TIMEOUT_MS"

_DEFAULT_FORCE_FLUSH_TIMEOUT_MS = 3000
_MAX_FORCE_FLUSH_TIMEOUT_MS = 60_000

# OpenTelemetry instrumentation scope name. Renaming it would split
# already-exported traces across two scopes, so it keeps the published
# name even though the package now lives under ``dashscope``.
_TRACER_NAME = "dashscope.applications.finetune.reinforcement"
# I/O preview limits for span attributes.
_IO_MAX_LEN = 128 * 1024  # 128KB per input/output preview
# Per-string limit applied during JSON-friendly conversion (before
# json.dumps()).
_SERIALIZE_MAX_STR_LEN = 128 * 1024  # 128KB per string field

# Request-local upstream trace linkage (used to decide whether FC traces can
# correlate with upstream RFT traces). Stored in contextvars so it propagates
# across async and threadpool offload (FuncManager uses copy_context()).
_UPSTREAM_TRACEPARENT_PRESENT: contextvars.ContextVar[
    bool
] = contextvars.ContextVar(
    "agentic_rl_upstream_traceparent_present",
    default=False,
)
_UPSTREAM_TRACE_ID: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar("agentic_rl_upstream_trace_id", default=None)

# Baggage keys — aligned with the Bailian Agentic RL Tracing specification.
# BaggageSpanProcessor propagates these keys as span attributes on every
# descendant span.
AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY = "bailian.agentic_rl.rollout_id"
AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY = "bailian.agentic_rl.sample_id"
AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY = "bailian.agentic_rl.attempt_id"

# Span name constants — aligned with the Bailian Agentic RL Tracing
# specification.
_ENTRY_SPAN_NAME = "invoke_agentic_rl"
_ROLLOUT_SPAN_PREFIX = "invoke_rollout"
_REWARD_SPAN_PREFIX = "invoke_reward"

_baggage_span_processor_lock = threading.Lock()
_baggage_span_processor_state: str = "pending"  # pending | installed | skipped
# pending  - not yet attempted
# installed - BaggageSpanProcessor successfully registered
# skipped  - permanently disabled (OTel SDK / baggage package unavailable)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("true", "1", "yes")


def is_tracing_enabled() -> bool:
    return _env_truthy(_ENV_ENABLE_TRAJECTORY_SHORT)


class ForceFlushMode(str, Enum):
    """When to force-flush spans to the exporter.

    - none:     never force_flush (rely on BatchSpanProcessor schedule)
    - shutdown: only force_flush during graceful shutdown hooks
    - request:  force_flush at the end of each request (highest overhead)
    """

    NONE = "none"
    SHUTDOWN = "shutdown"
    REQUEST = "request"


def _get_force_flush_mode() -> ForceFlushMode:
    """Return configured force-flush mode.

    Default is ``none`` to avoid introducing request-path latency coupling.
    This switch is intended for platform/internal runtime configuration.
    """

    raw = os.environ.get(_ENV_FORCE_FLUSH_MODE, "").strip().lower()
    if not raw:
        return ForceFlushMode.NONE
    for m in ForceFlushMode:
        if raw == m.value:
            return m
    logger.warning(
        "[OTel] Invalid %s=%r; valid: %s. Falling back to %s.",
        _ENV_FORCE_FLUSH_MODE,
        raw,
        [m.value for m in ForceFlushMode],
        ForceFlushMode.NONE.value,
    )
    return ForceFlushMode.NONE


def _get_force_flush_timeout_ms() -> int:
    raw = os.environ.get(_ENV_FORCE_FLUSH_TIMEOUT_MS, "").strip()
    if not raw:
        return _DEFAULT_FORCE_FLUSH_TIMEOUT_MS
    try:
        v = int(raw)
    except Exception:
        logger.warning(
            "[OTel] Invalid %s=%r; using default 3000ms.",
            _ENV_FORCE_FLUSH_TIMEOUT_MS,
            raw,
        )
        return _DEFAULT_FORCE_FLUSH_TIMEOUT_MS
    # Clamp to a sensible range so a bad env value doesn't hang shutdown.
    return max(0, min(v, _MAX_FORCE_FLUSH_TIMEOUT_MS))


def _should_force_flush(*, reason: str) -> bool:
    """Return True if force flush is enabled for the given reason."""
    mode = _get_force_flush_mode()
    if mode == ForceFlushMode.NONE:
        return False
    if mode == ForceFlushMode.SHUTDOWN:
        return reason == "shutdown"
    if mode == ForceFlushMode.REQUEST:
        return reason == "request"
    return False


def maybe_force_flush(*, reason: str) -> None:
    """Best-effort force flush spans based on config.

    - Never raises (silent degradation).
    - No-ops when tracing is disabled or OTel SDK is unavailable.
    - Uses ``AGENTIC_RL_FORCE_FLUSH_MODE`` and
      ``AGENTIC_RL_FORCE_FLUSH_TIMEOUT_MS``.
    """

    if not is_tracing_enabled():
        return
    if not _OTEL_AVAILABLE:
        return

    if not _should_force_flush(reason=reason):
        return

    try:
        from opentelemetry.sdk.trace import TracerProvider
    except Exception:
        return

    try:
        provider = otel_trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            return
        provider.force_flush(timeout_millis=_get_force_flush_timeout_ms())
    except Exception:
        # Keep silent to avoid impacting business flow / shutdown path.
        return


async def maybe_force_flush_async(*, reason: str) -> None:
    """Async wrapper around :func:`maybe_force_flush` for async servers.

    This avoids blocking the event loop when force flushing spans.
    It preserves silent degradation semantics and the same env-driven decision
    logic.
    """
    # Fast-path: if force flush would no-op, return quickly without offloading.
    if not is_tracing_enabled() or not _OTEL_AVAILABLE:
        return
    if not _should_force_flush(reason=reason):
        return

    try:
        await asyncio.to_thread(maybe_force_flush, reason=reason)
    except Exception:
        return


def _should_log_trace_id() -> bool:
    raw = os.environ.get("AGENTIC_RL_LOG_TRACE_ID", "false")
    return raw.strip().lower() not in ("0", "false", "no")


def current_trace_id_hex() -> Optional[str]:
    """Return the current OTel trace id as 32-char hex, or None.

    Returns None when there is no active span or OTel is not installed.
    """
    if not _OTEL_AVAILABLE:
        return None
    try:
        from opentelemetry.trace import format_trace_id

        ctx = otel_trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format_trace_id(ctx.trace_id)
    except Exception:
        pass
    return None


def current_span_ids_hex() -> Tuple[Optional[str], Optional[str]]:
    """Return trace/span id hex for the current span.

    Returns ``(None, None)`` when absent.
    """
    if not _OTEL_AVAILABLE:
        return None, None
    try:
        from opentelemetry.trace import format_span_id, format_trace_id

        ctx = otel_trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format_trace_id(ctx.trace_id), format_span_id(ctx.span_id)
    except Exception:
        pass
    return None, None


def set_upstream_trace_linkage(
    *,
    traceparent_present: bool,
    upstream_trace_id: Optional[str],
) -> Tuple[contextvars.Token, contextvars.Token]:
    """Set request-local upstream linkage flags for later span annotation."""
    t1 = _UPSTREAM_TRACEPARENT_PRESENT.set(bool(traceparent_present))
    t2 = _UPSTREAM_TRACE_ID.set(upstream_trace_id)
    return t1, t2


def reset_upstream_trace_linkage(
    tokens: Tuple[contextvars.Token, contextvars.Token],
) -> None:
    """Reset request-local upstream linkage flags."""
    t1, t2 = tokens
    _UPSTREAM_TRACEPARENT_PRESENT.reset(t1)
    _UPSTREAM_TRACE_ID.reset(t2)


def get_upstream_trace_linkage() -> Tuple[bool, Optional[str]]:
    """Return upstream linkage for the current request.

    Tuple is ``(traceparent_present, upstream_trace_id)``.
    """
    return _UPSTREAM_TRACEPARENT_PRESENT.get(), _UPSTREAM_TRACE_ID.get()


def log_trace_id(role: str) -> None:
    """
    Log the current ``trace_id`` at ``logger.info`` (stdout by default).

    Helps correlate with Console/OTLP. No-op unless
    ``AGENTIC_RL_LOG_TRACE_ID`` is truthy (default off).

    ``role`` should be a short label such as ``processor``,
    ``execute_tool:get_weather``, or ``llm``.
    """
    if not _should_log_trace_id():
        return
    tid, sid = current_span_ids_hex()
    logger.info(
        "agentic_rl observability trace_id=%s span_id=%s role=%s",
        tid or "(no active span)",
        sid or "(no active span)",
        role,
    )


def _truncate(s: str, max_len: int = _IO_MAX_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# pylint: disable-next=too-many-return-statements,too-many-branches
def _to_json_friendly(
    obj: Any,
) -> Any:
    """Recursively convert models/nested structures to JSON-serialisable
    natives.

    Suitable for span attributes (via ``json.dumps``).
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return _truncate(obj, _SERIALIZE_MAX_STR_LEN)
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, Dict):
        return {str(k): _to_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_friendly(x) for x in obj]
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump(mode="json")
        except Exception:
            try:
                dumped = obj.model_dump()
            except Exception:
                dumped = None
        if dumped is not None:
            return _to_json_friendly(dumped)
    if hasattr(obj, "dict"):
        try:
            dumped = obj.Dict()  # type: ignore[call-arg]
        except Exception:
            dumped = None
        if dumped is not None:
            return _to_json_friendly(dumped)
    return str(obj)


def _safe_json_preview(obj: Any) -> str:
    try:
        payload = _to_json_friendly(obj)
        return _truncate(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        return _truncate(repr(obj))


def span_payload_preview(obj: Any) -> str:
    """
    JSON-serialise ``obj`` for span attributes, truncated to ``_IO_MAX_LEN``.

    Used by ``observability.processor_span`` and similar modules in the same
    component package.
    """
    return _safe_json_preview(obj)


def get_tracer() -> Optional[Any]:
    """Return this library's OTel :class:`~opentelemetry.trace.Tracer`.

    Returns None when the SDK is not installed.
    """
    if not _OTEL_AVAILABLE:
        return None
    return otel_trace.get_tracer(_TRACER_NAME)


def ensure_agentic_rl_baggage_span_processor() -> None:
    """
    Ensure a real SDK :class:`~opentelemetry.sdk.trace.TracerProvider` is
    active and register
    :class:`~opentelemetry.processor.baggage.BaggageSpanProcessor` on it so
    baggage entries ``bailian.agentic_rl.rollout_id`` / ``sample_id`` /
    ``attempt_id`` are copied onto every new span as attributes (see
    :func:`rollout_context`).

    Designed to be called from FastAPI ``startup_event`` / ``lifespan``,
    which runs **inside** each uvicorn worker process after ``os.fork()``.
    This makes it safe to initialize
    :class:`~opentelemetry.sdk.trace.export.BatchSpanProcessor` (which spawns
    a background thread and opens a network connection) here — every worker
    gets its own independent exporter.

    If no real SDK TracerProvider is found, one is created automatically using
    the standard ``OTEL_*`` environment variables (endpoint, headers,
    protocol, resource attributes, etc.). ``service.instance.id`` is injected
    as ``worker-<pid>`` to distinguish per-worker spans in the backend. If
    ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` is absent, initialization is skipped
    and tracing is silently disabled for this worker.

    No-op if OpenTelemetry SDK / baggage processor packages are unavailable.
    """
    global _baggage_span_processor_state
    if _baggage_span_processor_state != "pending":
        return
    with _baggage_span_processor_lock:
        if _baggage_span_processor_state != "pending":
            return
        if not _OTEL_AVAILABLE:
            _baggage_span_processor_state = "skipped"
            return
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.processor.baggage import BaggageSpanProcessor
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError:  # pragma: no cover
            _baggage_span_processor_state = "skipped"
            return

        provider = ot_trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            # No real TracerProvider yet — create one using OTEL_* env vars.
            # This runs inside the forked worker process, so BatchSpanProcessor
            # background thread and exporter connection are fork-safe.
            endpoint = os.getenv(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            ) or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if not endpoint:
                # No endpoint: keep "pending" so a later external
                # set_tracer_provider() can still succeed.
                logger.debug(
                    "[OTel] OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set; "
                    "skipping TracerProvider init",
                )
                return
            try:
                from opentelemetry.exporter.otlp.proto.http import (
                    trace_exporter,
                )
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                OTLPHttpExporter = trace_exporter.OTLPSpanExporter

                # Resource.create() reads OTEL_SERVICE_NAME and
                # OTEL_RESOURCE_ATTRIBUTES automatically; we only inject
                # service.instance.id (pid) on top.
                resource = Resource.create(
                    {"service.instance.id": f"worker-{os.getpid()}"},
                )
                provider = TracerProvider(resource=resource)
                # Pass endpoint explicitly: OTLPSpanExporter auto-reads
                # OTEL_EXPORTER_OTLP_ENDPOINT but not
                # OTEL_EXPORTER_OTLP_TRACES_ENDPOINT; resolve it ourselves.
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPHttpExporter(endpoint=endpoint)),
                )
                ot_trace.set_tracer_provider(provider)
                logger.info(
                    "[OTel] TracerProvider initialized: endpoint=%s pid=%s",
                    endpoint,
                    os.getpid(),
                )
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "[OTel] Failed to initialize TracerProvider: %s",
                    exc,
                )
                _baggage_span_processor_state = "skipped"
                return

        def _allow_baggage_key(baggage_key: str) -> bool:
            return baggage_key in (
                AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY,
                AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY,
                AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY,
            )

        provider.add_span_processor(BaggageSpanProcessor(_allow_baggage_key))
        _baggage_span_processor_state = "installed"
        logger.info(
            "[OTel] BaggageSpanProcessor registered (pid=%s)",
            os.getpid(),
        )


@contextmanager
def rollout_context(
    rollout_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
) -> Iterator[None]:
    """
    Inject ``rollout_id`` / ``sample_id`` / ``attempt_id`` into W3C Baggage
    so that all descendant spans created inside the block automatically carry
    these three business IDs as span attributes (keys:
    ``bailian.agentic_rl.rollout_id`` / ``sample_id`` / ``attempt_id``).
    """
    if not _OTEL_AVAILABLE or not (rollout_id or sample_id or attempt_id):
        yield
        return
    try:
        from opentelemetry import baggage
        from opentelemetry import context as otel_context
    except ImportError:  # pragma: no cover
        yield
        return
    ctx = otel_context.get_current()
    if rollout_id:
        ctx = baggage.set_baggage(
            AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY,
            rollout_id,
            ctx,
        )
    if sample_id:
        ctx = baggage.set_baggage(
            AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY,
            sample_id,
            ctx,
        )
    if attempt_id:
        ctx = baggage.set_baggage(
            AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY,
            attempt_id,
            ctx,
        )
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def _extract_baggage_ids_from_input(
    input_data: BaseDataModel,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (rollout_id, sample_id, attempt_id) from input_data.

    Priority: request_metadata > task.rollout_id > input_data.rollout_id
    """
    rollout_id: Optional[str] = None
    sample_id: Optional[str] = None
    attempt_id: Optional[str] = None

    meta = getattr(input_data, "request_metadata", None)
    if meta is not None:
        v = getattr(meta, "rollout_id", None)
        if v:
            rollout_id = str(v)
        v = getattr(meta, "sample_id", None)
        if v:
            sample_id = str(v)
        v = getattr(meta, "attempt_id", None)
        if v:
            attempt_id = str(v)

    if not rollout_id:
        task = getattr(input_data, "task", None)
        if task is not None:
            v = getattr(task, "rollout_id", None)
            if v:
                rollout_id = str(v)

    if not rollout_id:
        v = getattr(input_data, "rollout_id", None)
        if v:
            rollout_id = str(v)

    return rollout_id, sample_id, attempt_id


def _rollout_id_from_input(input_data: BaseDataModel) -> Optional[str]:
    """Backward-compatible helper: return only rollout_id."""
    rid, _, _ = _extract_baggage_ids_from_input(input_data)
    return rid


def resolve_processor_func_type_for_span(
    input_data: BaseDataModel,
    logical_kind: str,
) -> FuncType:
    """
    Decorator-path helper: prefer ``input_data.func_type`` when available;
    otherwise map ``logical_kind`` (case-insensitive) to ``Reward`` /
    ``Rollout``.
    """
    ft = getattr(input_data, "func_type", None)
    if ft is not None:
        return ft
    lk = logical_kind.strip().lower()
    if lk == "reward":
        return FuncType.REWARD
    if lk == "rollout":
        return FuncType.ROLLOUT
    raise ValueError(
        f"Cannot resolve FunctionType for span: input has no func_type and "
        f"logical_kind={logical_kind!r} is not reward/rollout",
    )


def apply_processor_span_attributes_before(
    span: Any,
    *,
    func_type: FuncType,
    input_data: BaseDataModel,
    capture_full_io: bool = True,
) -> None:
    """
    Reward/Rollout processor span: input-side attributes for
    ``@observe_processor`` / ``trace_processor_span``.

    - ``gen_ai.span.kind`` is ``REWARD`` or ``ROLLOUT`` (uppercase).
    - ``operation.name`` is ``invoke_reward`` or ``invoke_rollout``
      (aligned with Bailian specification).
    - Traffic labels ``bailian.agentic_rl.rollout_id`` / ``sample_id`` /
      ``attempt_id`` are set on the span; BaggageSpanProcessor also copies
      them via Baggage.
    - ``input.value`` is truncated JSON of the input model unless
      ``capture_full_io=False``.
    """
    kind_upper = func_type.value.upper()  # "ROLLOUT" or "REWARD"
    span.set_attribute("gen_ai.span.kind", kind_upper)
    # operation.name mirrors the span-name prefix per Bailian specification
    op_name = (
        _ROLLOUT_SPAN_PREFIX
        if func_type == FuncType.ROLLOUT
        else _REWARD_SPAN_PREFIX
    )
    span.set_attribute("operation.name", op_name)
    rid, sid, aid = _extract_baggage_ids_from_input(input_data)
    if rid:
        span.set_attribute(AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY, rid)
        span.set_attribute("request_id", rid)  # backward-compatible alias
    if sid:
        span.set_attribute(AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY, sid)
    if aid:
        span.set_attribute(AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY, aid)
    if func_type == FuncType.ROLLOUT:
        resource = getattr(input_data, "resource", None)
        if resource is not None:
            model_name = getattr(resource, "model_name", None)
            if model_name:
                span.set_attribute("gen_ai.request.model", str(model_name))
    if capture_full_io:
        span.set_attribute("input.value", span_payload_preview(input_data))


def apply_processor_span_attributes_after(
    span: Any,
    result: Any,
    *,
    capture_full_io: bool = True,
) -> None:
    """Set output-side attributes: ``output.value`` and result summary.

    Consistent with ``_set_result_attributes``.
    """
    if capture_full_io and result is not None:
        span.set_attribute("output.value", span_payload_preview(result))
    _set_result_attributes(span, result)


def _set_result_attributes(span: Any, result: Any) -> None:
    if result is None:
        return
    status_val = getattr(result, "status", None)
    if status_val is not None:
        span.set_attribute(
            "agentic_rl.result.status",
            (
                status_val.value
                if hasattr(status_val, "value")
                else str(status_val)
            ),
        )
    latency = getattr(result, "latency", None)
    if latency is not None:
        span.set_attribute("agentic_rl.result.latency_s", float(latency))
    if hasattr(result, "reward") and result.reward is not None:
        score = getattr(result.reward, "reward_score", None)
        if score is not None:
            span.set_attribute("agentic_rl.result.reward_score", float(score))


def _set_entry_chain_span_attributes(
    span: Any,
    rid: Optional[str],
    sid: Optional[str],
    aid: Optional[str],
) -> None:
    """Populate the ``invoke_agentic_rl`` CHAIN span with id attributes.

    Only routing / identification attributes are written here; business-level
    I/O belongs exclusively to the child ROLLOUT/REWARD span produced by
    ``@observe_processor``.
    """
    span.set_attribute("gen_ai.span.kind", "CHAIN")
    span.set_attribute("operation.name", _ENTRY_SPAN_NAME)
    span.set_attribute("gen_ai.user.query_root_flag", 1)
    if rid:
        span.set_attribute(AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY, rid)
        span.set_attribute("request_id", rid)  # backward-compatible alias
    if sid:
        span.set_attribute(AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY, sid)
    if aid:
        span.set_attribute(AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY, aid)

    # Upstream linkage markers (for correlation with RFT).
    upstream_present, upstream_tid = get_upstream_trace_linkage()
    span.set_attribute("agentic_rl.trace.linked", 1 if upstream_present else 0)
    if upstream_tid:
        span.set_attribute("agentic_rl.upstream.trace_id", str(upstream_tid))


def trace_processor_process(
    _func_type: FuncType,
    processor: Any,
    input_data: BaseDataModel,
) -> Any:
    """
    FuncManager call path (sync processor): emit the root entry span
    (``invoke_agentic_rl``, ``gen_ai.span.kind=CHAIN``) whose main
    responsibility is injecting Baggage and serving as the trace root anchor
    for all downstream LLM/Tool spans.

    Full input/output/result attributes are intentionally omitted here — they
    belong to the ``@observe_processor`` decorator layer (ROLLOUT/REWARD
    spans).
    """
    if not is_tracing_enabled():
        return processor.process(input_data)

    if not _OTEL_AVAILABLE:
        logger.warning(
            "ENABLE_TRAJECTORY is set but "
            "OpenTelemetry is not installed. Reinstall the SDK with: "
            "pip install -U dashscope",
        )
        return processor.process(input_data)

    ensure_agentic_rl_baggage_span_processor()

    tracer = otel_trace.get_tracer(_TRACER_NAME)
    span_kw: Dict[str, Any] = {}
    if OtelSpanKind is not None:
        span_kw["kind"] = OtelSpanKind.INTERNAL

    rid, sid, aid = _extract_baggage_ids_from_input(input_data)
    with rollout_context(rollout_id=rid, sample_id=sid, attempt_id=aid):
        with tracer.start_as_current_span(_ENTRY_SPAN_NAME, **span_kw) as span:
            _set_entry_chain_span_attributes(span, rid, sid, aid)
            log_trace_id(f"processor:{_ENTRY_SPAN_NAME}")
            try:
                result = processor.process(input_data)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            span.set_status(Status(StatusCode.OK))
            return result


async def async_trace_processor_process(
    _func_type: FuncType,
    processor: Any,
    input_data: BaseDataModel,
) -> Any:
    """
    FuncManager call path (async processor): same logic as
    :func:`trace_processor_process`, but ``await``s ``processor.process``
    inside the entry span's ``with`` context so that child spans correctly
    inherit the parent span context.
    """
    if not is_tracing_enabled():
        return await processor.process(input_data)

    if not _OTEL_AVAILABLE:
        logger.warning(
            "ENABLE_TRAJECTORY is set but "
            "OpenTelemetry is not installed. Reinstall the SDK with: "
            "pip install -U dashscope",
        )
        return await processor.process(input_data)

    ensure_agentic_rl_baggage_span_processor()

    tracer = otel_trace.get_tracer(_TRACER_NAME)
    span_kw: Dict[str, Any] = {}
    if OtelSpanKind is not None:
        span_kw["kind"] = OtelSpanKind.INTERNAL

    rid, sid, aid = _extract_baggage_ids_from_input(input_data)
    with rollout_context(rollout_id=rid, sample_id=sid, attempt_id=aid):
        with tracer.start_as_current_span(_ENTRY_SPAN_NAME, **span_kw) as span:
            _set_entry_chain_span_attributes(span, rid, sid, aid)
            log_trace_id(f"processor:{_ENTRY_SPAN_NAME}")
            try:
                result = await processor.process(input_data)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            span.set_status(Status(StatusCode.OK))
            return result
