# -*- coding: utf-8 -*-
"""OpenTelemetry integration for AgenticRL function components.

This module provides a *stable public surface* while keeping heavy
observability dependencies lazily imported.  User-facing imports such as::

    from dashscope.finetune.reinforcement.component.observability import (
        observe_processor,
        trace_client,
        trace_tool,
    )

continue to work unchanged, but underlying GenAI / OTel helpers are only
imported on first attribute access, reducing cold-start and import-time
failure risk when features are unused.

For pylint / pyright in **library** code, prefer explicit submodule imports
(``...observability.genai``, ``...observability.processor_span``,
``...observability.tracing``) so static analysis matches the lazy surface.
"""

from __future__ import annotations

from typing import Any

# Public names are provided lazily via __getattr__; __all__ entries are not
# bindings.
# pylint: disable=undefined-all-variable
__all__ = [
    # public API
    "observe_llm",
    "observe_processor",
    "observe_tool",
    "rollout_context",
    "trace_client",
    "trace_tool",
    # utility functions / low-level interfaces
    "AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY",
    "AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY",
    "AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY",
    "apply_processor_span_attributes_after",
    "apply_processor_span_attributes_before",
    "current_trace_id_hex",
    "ensure_agentic_rl_baggage_span_processor",
    "dashscope_response_to_output_messages",
    "genai_llm_span",
    "get_tracer",
    "instrument_dashscope_generation_call",
    "instrument_openai_chat_completions",
    "is_tracing_enabled",
    "log_trace_id",
    "openai_chat_messages_to_input_messages",
    "openai_completion_to_output_messages",
    "run_with_genai_llm_span",
    "resolve_processor_func_type_for_span",
    "span_payload_preview",
    "trace_processor_process",
    "async_trace_processor_process",
    "trace_processor_span",
]
# pylint: enable=undefined-all-variable


def __getattr__(name: str) -> Any:
    """Lazy attribute resolver for the observability public surface.

    This keeps imports cheap for callers that only pull in a subset of the API,
    while preserving backwards-compatible symbols.
    """
    # GenAI / tool-related helpers
    if name in {
        "dashscope_response_to_output_messages",
        "genai_llm_span",
        "instrument_dashscope_generation_call",
        "instrument_openai_chat_completions",
        "observe_llm",
        "observe_tool",
        "openai_chat_messages_to_input_messages",
        "openai_completion_to_output_messages",
        "run_with_genai_llm_span",
        "trace_client",
        "trace_tool",
    }:
        from dashscope.finetune.reinforcement.component.observability import (
            genai as _genai,
        )

        return getattr(_genai, name)

    # Processor-span helpers
    if name in {"observe_processor", "trace_processor_span"}:
        from dashscope.finetune.reinforcement.component.observability import (
            processor_span as _processor_span,
        )

        return getattr(_processor_span, name)

    # Tracing core helpers
    if name in {
        "AGENTIC_RL_ATTEMPT_ID_BAGGAGE_KEY",
        "AGENTIC_RL_ROLLOUT_ID_BAGGAGE_KEY",
        "AGENTIC_RL_SAMPLE_ID_BAGGAGE_KEY",
        "apply_processor_span_attributes_after",
        "apply_processor_span_attributes_before",
        "async_trace_processor_process",
        "current_trace_id_hex",
        "ensure_agentic_rl_baggage_span_processor",
        "get_tracer",
        "is_tracing_enabled",
        "log_trace_id",
        "resolve_processor_func_type_for_span",
        "rollout_context",
        "span_payload_preview",
        "trace_processor_process",
    }:
        from dashscope.finetune.reinforcement.component.observability import (
            tracing as _tracing,
        )

        return getattr(_tracing, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
