# -*- coding: utf-8 -*-
"""
OTel span decorator for the ``process`` method of Reward / Rollout and
similar processors.

Emits a full business span (``gen_ai.span.kind=REWARD`` / ``ROLLOUT``) and
records input / output / result attributes.

Forms a parent-child hierarchy with the ``FuncManager`` CHAIN entry span
from ``tracing.trace_processor_process``: the CHAIN span handles Baggage
injection and dispatch tracking, while this decorator records business-level
semantics. Both can be used together.

Span naming convention (aligned with Bailian Agentic RL Tracing
specification):
- Rollout: ``invoke_rollout <ClassName>``
- Reward:  ``invoke_reward <ClassName>``

OpenTelemetry ``SpanKind`` is ``INTERNAL`` (overridable). Logical kind is
``gen_ai.span.kind`` (``REWARD`` / ``ROLLOUT``).

.. note::

    ``FuncManager(..., observe=True)`` (default) produces a lightweight CHAIN
    span (``gen_ai.span.kind=CHAIN``). The business span produced by this
    decorator (``REWARD`` / ``ROLLOUT``) is semantically its child; both can be
    used simultaneously. If ``FuncManager(..., observe=False)`` and this
    decorator is also absent, no span is emitted.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from dashscope.finetune.reinforcement.common.log import logger
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
)
from dashscope.finetune.reinforcement.component.observability.tracing import (
    _REWARD_SPAN_PREFIX,
    _ROLLOUT_SPAN_PREFIX,
    apply_processor_span_attributes_after,
    apply_processor_span_attributes_before,
    ensure_agentic_rl_baggage_span_processor,
    get_tracer,
    is_tracing_enabled,
    log_trace_id,
    resolve_processor_func_type_for_span,
)

try:
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _OTEL_SPAN_API = True
except ImportError:  # pragma: no cover
    SpanKind = None  # type: ignore[misc, assignment]
    Status = None  # type: ignore[misc, assignment]
    StatusCode = None  # type: ignore[misc, assignment]
    _OTEL_SPAN_API = False

F = TypeVar("F", bound=Callable[..., Any])


def trace_processor_span(  # pylint: disable=too-many-statements
    _fn: Optional[F] = None,
    *,
    logical_kind: str,
    otel_span_kind: Any = None,
    capture_io_attributes: bool = True,
) -> Any:
    """
    Generic decorator wrapping ``process(self, input: BaseDataModel, ...)``
    ``-> Any``.

    ``logical_kind`` is used to resolve the span kind to ``reward`` /
    ``rollout`` (case-insensitive) when ``input.func_type`` is absent;
    otherwise ``input_data.func_type`` takes precedence (consistent with
    ``FuncManager``).
    """

    def decorator(fn: F) -> F:  # pylint: disable=too-many-statements
        default_kind = (
            SpanKind.INTERNAL
            if _OTEL_SPAN_API and SpanKind is not None
            else None
        )
        kind_arg = (
            otel_span_kind if otel_span_kind is not None else default_kind
        )

        def _run_with_span(
            self: Any,
            input_data: BaseDataModel,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not is_tracing_enabled():
                return fn(self, input_data, *args, **kwargs)
            tracer = get_tracer()
            if tracer is None or not _OTEL_SPAN_API:
                if is_tracing_enabled() and tracer is None:
                    logger.warning(
                        "ENABLE_TRAJECTORY is set but "
                        "OpenTelemetry is not installed. "
                        "Reinstall the SDK with: pip install -U dashscope",
                    )
                return fn(self, input_data, *args, **kwargs)

            try:
                func_type = resolve_processor_func_type_for_span(
                    input_data,
                    logical_kind,
                )
            except ValueError as e:
                logger.warning("trace_processor_span: %s — skipping span", e)
                return fn(self, input_data, *args, **kwargs)

            ensure_agentic_rl_baggage_span_processor()
            # Span name: "invoke_rollout <ClassName>" or
            # "invoke_reward <ClassName>"
            prefix = (
                _ROLLOUT_SPAN_PREFIX
                if func_type.value.lower() == "rollout"
                else _REWARD_SPAN_PREFIX
            )
            span_name = f"{prefix} {type(self).__name__}"
            span_kw: Dict[str, Any] = {}
            if kind_arg is not None:
                span_kw["kind"] = kind_arg

            with tracer.start_as_current_span(span_name, **span_kw) as span:
                apply_processor_span_attributes_before(
                    span,
                    func_type=func_type,
                    input_data=input_data,
                    capture_full_io=capture_io_attributes,
                )
                log_trace_id(f"processor_span:{func_type.value}")
                try:
                    result = fn(self, input_data, *args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                apply_processor_span_attributes_after(
                    span,
                    result,
                    capture_full_io=capture_io_attributes,
                )
                span.set_status(Status(StatusCode.OK))
                return result

        async def _async_run_with_span(
            self: Any,
            input_data: BaseDataModel,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not is_tracing_enabled():
                return await fn(self, input_data, *args, **kwargs)
            tracer = get_tracer()
            if tracer is None or not _OTEL_SPAN_API:
                return await fn(self, input_data, *args, **kwargs)

            try:
                func_type = resolve_processor_func_type_for_span(
                    input_data,
                    logical_kind,
                )
            except ValueError as e:
                logger.warning("trace_processor_span: %s — skipping span", e)
                return await fn(self, input_data, *args, **kwargs)

            ensure_agentic_rl_baggage_span_processor()
            # Span name: "invoke_rollout <ClassName>" or
            # "invoke_reward <ClassName>"
            prefix = (
                _ROLLOUT_SPAN_PREFIX
                if func_type.value.lower() == "rollout"
                else _REWARD_SPAN_PREFIX
            )
            span_name = f"{prefix} {type(self).__name__}"
            span_kw: Dict[str, Any] = {}
            if kind_arg is not None:
                span_kw["kind"] = kind_arg

            with tracer.start_as_current_span(span_name, **span_kw) as span:
                apply_processor_span_attributes_before(
                    span,
                    func_type=func_type,
                    input_data=input_data,
                    capture_full_io=capture_io_attributes,
                )
                log_trace_id(f"processor_span:{func_type.value}")
                try:
                    result = await fn(self, input_data, *args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                apply_processor_span_attributes_after(
                    span,
                    result,
                    capture_full_io=capture_io_attributes,
                )
                span.set_status(Status(StatusCode.OK))
                return result

        if inspect.iscoroutinefunction(fn):
            return cast(F, wraps(fn)(_async_run_with_span))
        return cast(F, wraps(fn)(_run_with_span))

    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


def observe_processor(
    _fn: Optional[F] = None,
    *,
    otel_span_kind: Any = None,
    capture_io_attributes: bool = True,
) -> Any:
    """
    Processor decorator that infers ``func_type`` from the decorated class MRO.

    - Inherits ``AbstractRolloutProcessor`` → ``FuncType.ROLLOUT``
    - Inherits ``AbstractRewardProcessor``  → ``FuncType.REWARD``
    - Falls back to ``logical_kind="generic"`` on inference failure (no error
      raised)

    Replaces the explicit ``@trace_processor_span(logical_kind=...)`` style.
    """
    from dashscope.finetune.reinforcement.component.processor import (
        AbstractRewardProcessor,
        AbstractRolloutProcessor,
    )

    _KIND_MAP = {
        AbstractRolloutProcessor: "Rollout",
        AbstractRewardProcessor: "Reward",
    }

    def _infer_kind(instance: Any) -> str:
        for base_cls, kind in _KIND_MAP.items():
            if isinstance(instance, base_cls):
                return kind
        return "generic"

    def decorator(fn: F) -> F:
        # Pre-compile wrappers for two fixed paths:
        # 1. has_func_type path: logical_kind is unused (resolve reads
        #    input.func_type directly)
        # 2. inferred path: the kind for each subclass is fixed and could be
        #    pre-compiled, but since it depends on the type of self (unknown at
        #    decoration time), one cached wrapper per "Reward" / "Rollout" /
        #    "generic" is kept instead.
        _cached: Dict[str, Any] = {}

        def _get_wrapped(kind: str) -> Any:
            if kind not in _cached:
                _cached[kind] = trace_processor_span(
                    fn,
                    logical_kind=kind,
                    otel_span_kind=otel_span_kind,
                    capture_io_attributes=capture_io_attributes,
                )
            return _cached[kind]

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def auto_wrapper(
                self: Any,
                input_data: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                has_func_type = (
                    getattr(input_data, "func_type", None) is not None
                )
                kind = (
                    "reward_or_rollout" if has_func_type else _infer_kind(self)
                )
                return await _get_wrapped(kind)(
                    self,
                    input_data,
                    *args,
                    **kwargs,
                )

            return auto_wrapper  # type: ignore[return-value]
        else:

            @wraps(fn)
            # type: ignore[misc]
            def auto_wrapper(
                self: Any,
                input_data: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                has_func_type = (
                    getattr(input_data, "func_type", None) is not None
                )
                kind = (
                    "reward_or_rollout" if has_func_type else _infer_kind(self)
                )
                return _get_wrapped(kind)(self, input_data, *args, **kwargs)

            return auto_wrapper  # type: ignore[return-value]

    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


__all__ = [
    "observe_processor",
    "trace_processor_span",
]
