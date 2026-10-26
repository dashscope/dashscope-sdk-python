# -*- coding: utf-8 -*-
"""
GenAI span helpers (LoongSuite / ``loongsuite-util-genai``).

Submodules by responsibility:

- :mod:`~dashscope.finetune.reinforcement.component.observability.genai.messages`  # noqa: E501
  — message and tool_calls mapping
- :mod:`~dashscope.finetune.reinforcement.component.observability.genai.llm_openai`  # noqa: E501
  — OpenAI-compatible LLM client patching
- :mod:`~dashscope.finetune.reinforcement.component.observability.genai.llm_dashscope`  # noqa: E501
  — DashScope Generation patching
- :mod:`~dashscope.finetune.reinforcement.component.observability.genai.tools`
  — tool tracing (decorator + patcher)

Requires ``pip install loongsuite-util-genai`` and tracing env flags
(``ENABLE_TRAJECTORY``).

Optional: ``AGENTIC_RL_LOG_TRACE_ID=true`` enables ``trace_id`` diagnostic
logs (default off);
see :mod:`dashscope.finetune.reinforcement.component.observability.tracing`.
"""

from __future__ import annotations

from typing import Any

from dashscope.finetune.reinforcement.component.observability.genai.llm_dashscope import (  # noqa: E501  # pylint: disable=line-too-long
    instrument_dashscope_generation_call,
)
from dashscope.finetune.reinforcement.component.observability.genai.llm_openai import (  # noqa: E501  # pylint: disable=line-too-long
    instrument_openai_chat_completions,
)
from dashscope.finetune.reinforcement.component.observability.genai.llm_span import (  # noqa: E501  # pylint: disable=line-too-long
    genai_llm_span,
    observe_llm,
    run_with_genai_llm_span,
)
from dashscope.finetune.reinforcement.component.observability.genai.messages import (  # noqa: E501  # pylint: disable=line-too-long
    dashscope_response_to_output_messages,
    openai_chat_messages_to_input_messages,
    openai_completion_to_output_messages,
)
from dashscope.finetune.reinforcement.component.observability.genai.tools import (  # noqa: E501  # pylint: disable=line-too-long
    observe_tool,
    trace_tool,
)


def trace_client(client: Any) -> None:
    """
    Enable tracing on an LLM client so that every call automatically produces
    an LLM span.

    Supported shapes (detected by structure, not by class name):

    * **Full OpenAI client** — ``openai.OpenAI`` / ``openai.AsyncOpenAI``
    instances that expose ``.chat.completions.create``.
    * **Completions resource** — objects that have a ``.create`` method but no
      ``.chat`` attribute (e.g. ``ChatOpenAI.client`` from
      ``langchain-openai``).
    * **LangChain-like LLM wrapper** — objects with ``.client`` and/or
    ``.async_client`` attributes that point to completions resources. Both
    sync and async resources are patched if present.
    * **DashScope Generation class** — the class itself (not an instance),
    which has a ``call`` classmethod.

    The function is idempotent — patching the same object twice is safe.
    No-op when tracing is disabled or ``loongsuite-util-genai`` is not
    installed.

    .. note::
        For DashScope, pass the class (``Generation``), not an instance.
        For OpenAI / LangChain, pass an instance.
    """
    import inspect as _inspect

    # 1. Full OpenAI client: has .chat.completions.create
    chat = getattr(client, "chat", None)
    completions = (
        getattr(chat, "completions", None) if chat is not None else None
    )
    if completions is not None and callable(
        getattr(completions, "create", None),
    ):
        instrument_openai_chat_completions(client)
        return

    # 2. Completions resource directly: has .create but no .chat
    if callable(getattr(client, "create", None)) and not hasattr(
        client,
        "chat",
    ):
        instrument_openai_chat_completions(client)
        return

    # 3. LangChain-like wrapper: has .client and/or .async_client pointing
    # to completions
    sync_client = getattr(client, "client", None)
    async_client = getattr(client, "async_client", None)
    if sync_client is not None or async_client is not None:
        if sync_client is not None and callable(
            getattr(sync_client, "create", None),
        ):
            instrument_openai_chat_completions(sync_client)
        if async_client is not None and callable(
            getattr(async_client, "create", None),
        ):
            instrument_openai_chat_completions(async_client)
        return

    # 4. DashScope Generation class: has a classmethod `call`
    candidate = client if _inspect.isclass(client) else type(client)
    has_call_classmethod = _inspect.isclass(candidate) and isinstance(
        candidate.__dict__.get("call"),
        classmethod,
    )
    if has_call_classmethod:
        instrument_dashscope_generation_call(candidate)
        return

    # Unsupported client type: silent degradation (observability should
    # never break business logic)
    import logging

    logging.getLogger(__name__).warning(
        "trace_client: unsupported client type %r. Skipping tracing. "
        "Expected an OpenAI client, LangChain-like wrapper, or DashScope "
        "Generation class.",
        type(client),
    )


__all__ = [
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
]
