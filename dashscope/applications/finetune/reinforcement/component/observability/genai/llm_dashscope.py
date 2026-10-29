# -*- coding: utf-8 -*-
"""DashScope native ``Generation.call`` class-method instrumentation."""
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from opentelemetry.trace.status import Status, StatusCode

from ....common.log import logger
from ..genai._core import (  # noqa: E501
    GENAI_AVAILABLE,
    get_handler,
)
from ..genai.messages import (  # noqa: E501  # pylint: disable=line-too-long
    dashscope_response_to_output_messages,
    openai_chat_messages_to_input_messages,
)
from ..tracing import (  # noqa: E501
    is_tracing_enabled,
    log_trace_id,
)

# Idempotency sentinel on the patched class — owned by this module (cf.
# ``trace_tool``).
_DASHSCOPE_GENAI_PATCHED_ATTR = "_agentic_rl_genai_dashscope_patched"


def _fill_llm_invocation_from_dashscope_kwargs(
    inv: Any,
    model: Any,
    prompt: Any,
    _history: Any,
    messages: Any,
    kwargs: Dict[str, Any],
    provider: str,
) -> None:
    inv.provider = provider
    inv.request_model = model
    if messages is not None and len(messages) > 0:
        inv.input_messages = openai_chat_messages_to_input_messages(messages)
    elif prompt is not None and prompt != "":
        inv.input_messages = openai_chat_messages_to_input_messages(
            [{"role": "user", "content": prompt}],
        )
    # ``history`` (multi-turn context) is forwarded to the underlying call
    # but not mapped to GenAI span attributes — DashScope treats it as part
    # of prompt context.
    inv.temperature = kwargs.get("temperature")
    inv.top_p = kwargs.get("top_p")
    inv.max_tokens = kwargs.get("max_tokens")
    stop = kwargs.get("stop")
    if stop is not None:
        inv.stop_sequences = [stop] if isinstance(stop, str) else list(stop)
    inv.seed = kwargs.get("seed")
    freq = kwargs.get("frequency_penalty")
    if freq is not None:
        inv.frequency_penalty = freq
    pres = kwargs.get("presence_penalty")
    if pres is not None:
        inv.presence_penalty = pres


def _apply_dashscope_response_to_invocation(
    inv: Any,
    response: Any,
    *,
    request_model: Any = None,
) -> None:
    inv.response_id = getattr(response, "request_id", None)
    inv.response_model_name = getattr(response, "model", None) or request_model
    usage = getattr(response, "usage", None)
    if usage is not None:
        inv.input_tokens = getattr(usage, "input_tokens", None) or getattr(
            usage,
            "prompt_tokens",
            None,
        )
        inv.output_tokens = getattr(usage, "output_tokens", None) or getattr(
            usage,
            "completion_tokens",
            None,
        )
    inv.output_messages = dashscope_response_to_output_messages(response)


def instrument_dashscope_generation_call(
    generation_cls: Any,
    *,
    provider: str = "dashscope",
    handler: Any = None,
) -> Any:
    """
    Attach a GenAI ``llm`` span to ``Generation.call`` (a class method).

    Complements :func:`instrument_openai_chat_completions`: that function
    targets OpenAI-compatible clients, while this one targets the DashScope
    native protocol. ``stream=True`` calls are passed through transparently
    to avoid mis-spanning generators.
    """
    if not is_tracing_enabled():
        return generation_cls
    if not GENAI_AVAILABLE:
        logger.warning(
            "GenAI tracing requested but loongsuite-util-genai is not "
            "installed. pip install loongsuite-util-genai",
        )
        return generation_cls
    # pylint: disable=protected-access
    if getattr(generation_cls, "_agentic_rl_genai_dashscope_patched", False):
        return generation_cls

    raw = generation_cls.__dict__.get("call")
    if raw is None or not isinstance(raw, classmethod):
        logger.warning(
            "instrument_dashscope_generation_call: %r has no classmethod "
            "'call'",
            generation_cls,
        )
        return generation_cls

    orig_fn = raw.__func__
    h = handler if handler is not None else get_handler()

    def wrapped_call(
        cls: Any,
        model: Any = None,
        prompt: Any = None,
        history: Any = None,
        api_key: Any = None,
        messages: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Pass through without a span when tracing is off, the GenAI library
        # is absent, or the call is a streaming request (spanning a
        # generator would mis-attribute latency).
        if (
            not is_tracing_enabled()
            or not GENAI_AVAILABLE
            or kwargs.get("stream")
        ):
            return orig_fn(
                cls,
                model,
                prompt=prompt,
                history=history,
                api_key=api_key,
                messages=messages,
                **kwargs,
            )
        with h.llm() as inv:
            log_trace_id("llm")
            _fill_llm_invocation_from_dashscope_kwargs(
                inv,
                model,
                prompt,
                history,
                messages,
                kwargs,
                provider,
            )
            try:
                response = orig_fn(
                    cls,
                    model,
                    prompt=prompt,
                    history=history,
                    api_key=api_key,
                    messages=messages,
                    **kwargs,
                )
                _apply_dashscope_response_to_invocation(
                    inv,
                    response,
                    request_model=model,
                )
                # Set OK status on successful execution
                if hasattr(inv, "span") and inv.span is not None:
                    inv.span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                # Set ERROR status on exception
                if hasattr(inv, "span") and inv.span is not None:
                    inv.span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    generation_cls.call = classmethod(wrapped_call)  # type: ignore[assignment]
    # pylint: disable=protected-access
    generation_cls._agentic_rl_genai_dashscope_patched = True
    return generation_cls
