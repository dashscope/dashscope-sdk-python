# -*- coding: utf-8 -*-
"""Manual / decorator-based ``llm`` span (for use without client
monkey-patching)."""
# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

from opentelemetry.trace.status import Status, StatusCode

from ..genai._core import (  # noqa: E501
    GENAI_AVAILABLE,
    get_handler,
)
from ..genai.messages import (  # noqa: E501  # pylint: disable=line-too-long
    dashscope_response_to_output_messages,
    openai_chat_messages_to_input_messages,
    openai_completion_to_output_messages,
)
from ..tracing import (
    is_tracing_enabled,
)

F = TypeVar("F", bound=Callable[..., Any])


def _set_span_ok(inv: Any) -> None:
    """Set OK status on span if available."""
    if hasattr(inv, "span") and inv.span is not None:
        inv.span.set_status(Status(StatusCode.OK))


def _set_span_error(inv: Any, error: Exception) -> None:
    """Set ERROR status on span if available."""
    if hasattr(inv, "span") and inv.span is not None:
        inv.span.set_status(Status(StatusCode.ERROR, str(error)))


def _apply_result_and_status(
    inv: Any,
    result: Any,
    result_mapper: Optional[Callable[[Any, Any], None]],
) -> None:
    """Apply result to invocation and set OK status."""
    if result_mapper is not None:
        result_mapper(inv, result)
    else:
        _apply_invocation_from_result_auto(inv, result)
    _set_span_ok(inv)


@contextmanager
def genai_llm_span(handler: Any = None) -> Iterator[Any]:
    if not is_tracing_enabled() or not GENAI_AVAILABLE:
        yield SimpleNamespace()
        return
    h = handler if handler is not None else get_handler()
    with h.llm() as inv:
        yield inv


def _fill_invocation_from_kwargs(
    inv: Any,
    *,
    provider: Optional[str],
    kwargs: Dict[str, Any],
    request_model_arg: str,
    messages_arg: str,
    prompt_arg: str,
) -> None:
    if provider:
        inv.provider = provider
    inv.request_model = kwargs.get(request_model_arg)
    messages = kwargs.get(messages_arg)
    if messages is not None:
        inv.input_messages = openai_chat_messages_to_input_messages(messages)
    else:
        prompt = kwargs.get(prompt_arg)
        if prompt is not None and prompt != "":
            inv.input_messages = openai_chat_messages_to_input_messages(
                [{"role": "user", "content": prompt}],
            )


def _apply_invocation_from_result_auto(inv: Any, result: Any) -> None:
    if result is None:
        return
    usage = getattr(result, "usage", None)
    if usage is not None:
        inv.input_tokens = getattr(usage, "prompt_tokens", None) or getattr(
            usage,
            "input_tokens",
            None,
        )
        inv.output_tokens = getattr(
            usage,
            "completion_tokens",
            None,
        ) or getattr(usage, "output_tokens", None)

    # OpenAI completion-like
    if hasattr(result, "choices"):
        inv.response_model_name = getattr(result, "model", None)
        inv.response_id = getattr(result, "id", None)
        inv.output_messages = openai_completion_to_output_messages(result)
        return

    # DashScope GenerationResponse-like
    if hasattr(result, "output"):
        inv.response_model_name = (
            getattr(result, "model", None) or inv.request_model
        )
        inv.response_id = getattr(result, "request_id", None)
        inv.output_messages = dashscope_response_to_output_messages(result)


def observe_llm(
    _fn: Optional[F] = None,
    *,
    provider: Optional[str] = None,
    request_model_arg: str = "model",
    messages_arg: str = "messages",
    prompt_arg: str = "prompt",
    result_mapper: Optional[Callable[[Any, Any], None]] = None,
    handler: Any = None,
) -> Any:
    """
    Wrap any LLM call function as an ``llm`` span (decorator style).

    By default, extracts input/output using common OpenAI/DashScope
    parameter names:
    - Input:  ``model`` / ``messages`` / ``prompt``
    - Output: auto-detected as OpenAI completion or DashScope
    GenerationResponse
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_tracing_enabled() or not GENAI_AVAILABLE:
                return fn(*args, **kwargs)
            h = handler if handler is not None else get_handler()
            with h.llm() as inv:
                _fill_invocation_from_kwargs(
                    inv,
                    provider=provider,
                    kwargs=kwargs,
                    request_model_arg=request_model_arg,
                    messages_arg=messages_arg,
                    prompt_arg=prompt_arg,
                )
                try:
                    result = fn(*args, **kwargs)
                    _apply_result_and_status(inv, result, result_mapper)
                    return result
                except Exception as e:
                    _set_span_error(inv, e)
                    raise

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_tracing_enabled() or not GENAI_AVAILABLE:
                return await fn(*args, **kwargs)
            h = handler if handler is not None else get_handler()
            with h.llm() as inv:
                _fill_invocation_from_kwargs(
                    inv,
                    provider=provider,
                    kwargs=kwargs,
                    request_model_arg=request_model_arg,
                    messages_arg=messages_arg,
                    prompt_arg=prompt_arg,
                )
                try:
                    result = await fn(*args, **kwargs)
                    _apply_result_and_status(inv, result, result_mapper)
                    return result
                except Exception as e:
                    _set_span_error(inv, e)
                    raise

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


def run_with_genai_llm_span(
    fn: Callable[..., Any],
    /,
    *fn_args: Any,
    provider: Optional[str] = None,
    request_model: Any = None,
    input_messages: Any = None,
    result_mapper: Optional[Callable[[Any, Any], None]] = None,
    handler: Any = None,
    **fn_kwargs: Any,
) -> Any:
    """
    Execute an LLM call and wrap it in an ``llm`` span (functional style).

    Suitable when you do not want to modify function signatures or
    permanently patch the client.
    """
    if not is_tracing_enabled() or not GENAI_AVAILABLE:
        return fn(*fn_args, **fn_kwargs)
    h = handler if handler is not None else get_handler()
    with h.llm() as inv:
        if provider:
            inv.provider = provider
        if request_model is not None:
            inv.request_model = request_model
        if input_messages is not None:
            inv.input_messages = openai_chat_messages_to_input_messages(
                input_messages,
            )
        try:
            result = fn(*fn_args, **fn_kwargs)
            _apply_result_and_status(inv, result, result_mapper)
            return result
        except Exception as e:
            _set_span_error(inv, e)
            raise


__all__ = [
    "genai_llm_span",
    "observe_llm",
    "run_with_genai_llm_span",
]
