# -*- coding: utf-8 -*-
"""
smoke_observability.py

Runnable smoke script (NOT a pytest test).

Goal: validate the **public** observability APIs exposed in `UserGuide.md`:
- `@observe_processor`: processor-level span
- `@observe_llm`: LLM span
- `@observe_tool`: tool span
- `trace_client(...)`: OpenAI-compatible client instrumentation (one
canonical shape)
- `trace_tool(...)`: tool instrumentation (one canonical shape)

Prerequisites (example):
  pip install -e .
  pip install loongsuite-util-genai

Run:
  ENABLE_TRAJECTORY=true python smoke_observability.py

Notes:
- This script uses `InMemorySpanExporter` to print spans locally. This is
for demo/smoke only. If you see spans printed here but nothing shows up in
ARMS, that's expected: this script does not export to ARMS/OTLP. Follow
`UserGuide.md` to configure tracing export for production runs. Do NOT copy
the `TracerProvider` / `set_tracer_provider(...)` part into your business
processors (e.g. `rollout.py` / `reward.py`). In production, configure
tracing via the SDK switch / standard OpenTelemetry exporters (
OTLP/Collector), rather than setting a global provider in each module.
- In some environments, optional tracing dependencies may segfault at
interpreter shutdown. To keep this script reliably runnable, it performs a
hard exit at the end.
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from dashscope.finetune.reinforcement.common.model_types import (
    FunctionType as FuncType,
)
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    AgentOutput,
    ModelResource,
)
from dashscope.finetune.reinforcement.component.data.reward_input import (
    RewardInput,
)
from dashscope.finetune.reinforcement.component.data.rollout_input import (
    RolloutInput,
)
from dashscope.finetune.reinforcement.component.func_manager import FuncManager
from dashscope.finetune.reinforcement.component.observability.genai import (
    observe_llm,
    observe_tool,
    trace_client,
    trace_tool,
)
from dashscope.finetune.reinforcement.component.observability.processor_span import (  # noqa: E501  # pylint: disable=line-too-long
    observe_processor,
)
from dashscope.finetune.reinforcement.component.processor.abstract_reward_processor import (  # noqa: E501  # pylint: disable=line-too-long
    AbstractRewardProcessor,
)
from dashscope.finetune.reinforcement.component.processor.abstract_rollout_processor import (  # noqa: E501  # pylint: disable=line-too-long
    AbstractRolloutProcessor,
)

# -----------------------------------------------------------------------------
# Local OTel setup (print spans via in-memory exporter)
# -----------------------------------------------------------------------------

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
otel_trace.set_tracer_provider(_provider)

_QUIET = False


def _clear_spans() -> None:
    _exporter.clear()


def _finished_spans() -> Sequence[Any]:
    return _exporter.get_finished_spans()


def _log(msg: str) -> None:
    if not _QUIET:
        print(msg)


def _summarize_span(span: Any) -> str:
    attrs = dict(span.attributes or {})
    kind = attrs.get("gen_ai.span.kind")

    # Keep output customer-friendly: only a few meaningful fields.
    interesting: Dict[str, Any] = {}
    for k in (
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.tool.name",
    ):
        if k in attrs:
            interesting[k] = attrs[k]

    extra = ""
    if interesting:
        extra = " " + " ".join([f"{k}={v!r}" for k, v in interesting.items()])
    return f"name={span.name!r} kind={kind!r}{extra}"


def _print_spans(tag: str) -> bool:
    spans = _finished_spans()
    if not spans:
        _log(f"  [{tag}] FAIL: no spans exported")
        return False
    _log(f"  [{tag}] PASS: spans={len(spans)}")
    for s in spans:
        _log(f"    - {_summarize_span(s)}")
    return True


async def _await_fm_process(fm: FuncManager, input_data: Any) -> Any:
    # FuncManager.process is async in this SDK version.
    return await fm.process(input_data)


# -----------------------------------------------------------------------------
# Mock inputs/processors
# -----------------------------------------------------------------------------


def _make_reward_input(*, rollout_id: str) -> RewardInput:
    return RewardInput(
        rollout_id=rollout_id,
        agent_output=AgentOutput(
            message=[{"role": "assistant", "content": "answer"}],
            reward_score=None,
        ),
        ground_truth="expected answer",
    )


def _make_rollout_input(*, rollout_id: str) -> RolloutInput:
    return RolloutInput(
        rollout_id=rollout_id,
        messages=[{"role": "user", "content": "What is 1+1?"}],
        model_resource=ModelResource(
            model_name="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-mock-key",
        ),
    )


class _SmokeRewardProcessor(AbstractRewardProcessor):
    @observe_processor
    def process(
        self,
        input_data: RewardInput,
    ) -> Any:  # pylint: disable=redefined-builtin
        return {"reward": 1.0, "rollout_id": input_data.rollout_id}


class _SmokeRolloutProcessor(AbstractRolloutProcessor):
    # pylint: disable=invalid-overridden-method
    @observe_processor
    def process(
        self,
        input_data: RolloutInput,
    ) -> Any:  # pylint: disable=redefined-builtin
        return {
            "trajectory": ["step1"],
            "rollout_id": getattr(input_data, "rollout_id", "ro-unknown"),
        }


# -----------------------------------------------------------------------------
# Smoke checks (minimal public API surface)
# -----------------------------------------------------------------------------


def smoke_observe_processor_reward() -> bool:
    _log("\n=== 1) @observe_processor (Reward) emits span ===")
    fm = FuncManager(
        FuncType.REWARD,
        processor=_SmokeRewardProcessor(),
        observe=False,
    )
    _ = asyncio.run(
        _await_fm_process(
            fm,
            _make_reward_input(rollout_id="ro-smoke-reward"),
        ),
    )
    ok = _print_spans("REWARD")
    _clear_spans()
    return ok


def smoke_observe_processor_rollout() -> bool:
    _log("\n=== 2) @observe_processor (Rollout) emits span ===")
    fm = FuncManager(
        FuncType.ROLLOUT,
        processor=_SmokeRolloutProcessor(),
        observe=False,
    )
    _ = asyncio.run(
        _await_fm_process(
            fm,
            _make_rollout_input(rollout_id="ro-smoke-rollout"),
        ),
    )
    ok = _print_spans("ROLLOUT")
    _clear_spans()
    return ok


def smoke_observe_llm() -> bool:
    _log("\n=== 3) @observe_llm emits LLM span ===")

    @observe_llm(
        provider="mock-provider",
        request_model_arg="model",
        messages_arg="messages",
    )
    def _call_llm(*, model: str, messages: List[Dict[str, str]]) -> Any:
        _ = messages
        choice = SimpleNamespace(
            message=SimpleNamespace(content="42", role="assistant"),
            finish_reason="stop",
        )
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2)
        return SimpleNamespace(
            model=model,
            id="mock-id",
            choices=[choice],
            usage=usage,
        )

    _ = _call_llm(
        model="mock-model",
        messages=[{"role": "user", "content": "What is 1+1?"}],
    )
    ok = _print_spans("LLM")
    _clear_spans()
    return ok


def smoke_observe_tool() -> bool:
    _log("\n=== 4) @observe_tool emits Tool span ===")

    @observe_tool(name="search_web")
    def search_web(query: str) -> str:
        return f"results for: {query}"

    _ = search_web("OpenTelemetry smoke")
    ok = _print_spans("TOOL")
    _clear_spans()
    return ok


def smoke_trace_client_openai_shape() -> bool:
    _log(
        "\n=== 5) trace_client instruments OpenAI-compatible client ("
        "chat.completions.create) ===",
    )

    class _Completions:
        def create(self, **_kwargs: Any) -> Any:
            msg = SimpleNamespace(role="assistant", content="hello")
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            return SimpleNamespace(
                model="fake-model",
                id="fake-id",
                choices=[choice],
                usage=usage,
            )

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    client = _Client()
    trace_client(client)
    _ = client.chat.completions.create(
        model="gpt-fake",
        messages=[{"role": "user", "content": "hi"}],
    )
    ok = _print_spans("trace_client")
    _clear_spans()
    return ok


def smoke_trace_tool_single() -> bool:
    _log("\n=== 6) trace_tool instruments a BaseTool-like object ===")

    class _Tool:
        name = "single_tool"

        def invoke(
            self,
            tool_input: Any,
            _config: Any = None,
            **_kwargs: Any,
        ) -> Dict[str, Any]:
            return {"ok": True, "input": tool_input}

        async def ainvoke(
            self,
            tool_input: Any,
            _config: Any = None,
            **_kwargs: Any,
        ) -> Dict[str, Any]:
            await asyncio.sleep(0)
            return {"ok": True, "input": tool_input}

    tool = _Tool()
    trace_tool(tool)

    # Call both paths to make failures obvious in either sync or async usage.
    _ = tool.invoke({"q": "ping"})
    asyncio.run(tool.ainvoke({"q": "ping"}))

    ok = _print_spans("trace_tool")
    _clear_spans()
    return ok


def _is_tracing_enabled() -> bool:
    v = os.environ.get("ENABLE_TRAJECTORY", "")
    return v.lower() in ("true", "1", "yes")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final PASS/FAIL summary.",
    )
    return p.parse_args(list(argv))


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    _QUIET = bool(args.quiet)
    if _QUIET:
        # Suppress SDK info logs in quiet mode.
        logging.getLogger("dashscope").setLevel(logging.WARNING)

    if not _is_tracing_enabled():
        _log("ENABLE_TRAJECTORY is not true -> tracing is OFF")
        _log("Run: ENABLE_TRAJECTORY=true python smoke_observability.py")
        sys.exit(0)

    _log("✓ ENABLE_TRAJECTORY=true, tracing is ON")

    checks = [
        ("observe_processor_reward", smoke_observe_processor_reward),
        ("observe_processor_rollout", smoke_observe_processor_rollout),
        ("observe_llm", smoke_observe_llm),
        ("observe_tool", smoke_observe_tool),
        ("trace_client_openai_shape", smoke_trace_client_openai_shape),
        ("trace_tool_single", smoke_trace_tool_single),
    ]

    passed = 0
    for _, fn in checks:
        try:
            if fn():
                passed += 1
        except Exception as e:
            _log(f"  [EXCEPTION] {type(e).__name__}: {e}")
            _clear_spans()

    total = len(checks)
    print(
        f"\nRESULT: {'PASS' if passed == total else 'FAIL'} ({passed}"
        f"/{total})",
        flush=True,
    )
    # Some optional tracing dependencies may segfault at interpreter shutdown
    # (outside of this SDK). Hard-exit keeps this script usable as a smoke
    # check.
    os._exit(0)
