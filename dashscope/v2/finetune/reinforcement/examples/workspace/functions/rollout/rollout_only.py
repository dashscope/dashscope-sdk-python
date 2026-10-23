# -*- coding: utf-8 -*-
"""
examples/workspace/functions/rollout/simple_rollout.py

Demo implementation of a Rollout Processor.
Demonstrates how to construct a simple rollout execution flow (simulating
Agent invocation).

Observability coverage:
- @observe_processor : ENTRY-level Span for the whole rollout
    (auto-inferred as ROLLOUT)
- @observe_llm       : LLM-level Span wrapping the simulated model call (
_call_llm)
- @observe_tool      : Tool-level Span wrapping tool calls
- trace_client       : Exercises all four routing branches via mock clients:
                        1. Full OpenAI client (.chat.completions.create)
                        2. Direct completions object (.create, no .chat)
                        3. LangChain-like wrapper (.client / .async_client)
                        4. DashScope Generation-like (classmethod call)
- trace_tool         : Exercises all input shapes + MCP auto-detection:
                        1. Single BaseTool
                        2. list[BaseTool]
                        3. tuple[BaseTool]
                        4. Dict[str, BaseTool]
                        5. ToolNode (LangGraph)
                        6. MCP tool (auto-detected provider)
                        7. Custom provider
                        8. Unsupported object (warning)
                        9. Idempotency check
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from dashscope.finetune.reinforcement import RolloutInput, RolloutOutput
from dashscope.finetune.reinforcement import logger
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    AgentOutput,
    TaskStatus,
)

# pylint: disable=no-name-in-module
from dashscope.finetune.reinforcement.component.observability import (
    observe_processor,
    observe_llm,
    observe_tool,
    trace_client,
    trace_tool,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRolloutProcessor,
)


# ========================================================================== #
# Mock clients for trace_client routing coverage                             #
# ========================================================================== #


def _make_mock_response(
    content: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> SimpleNamespace:
    """Factory for OpenAI-shaped mock responses."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            ),
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        model=model,
    )


# 1. Full OpenAI client: .chat.completions.create (sync and async variants)


class _MockSyncCompletions:
    """Sync completions resource attached to .chat.completions."""

    # pylint: disable=unused-argument
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        return _make_mock_response(
            "[mock] sync openai client",
            model,
            5,
            3,
        )


class _MockAsyncCompletions:
    """Async completions resource attached to .chat.completions."""

    # pylint: disable=unused-argument
    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        await asyncio.sleep(0)
        return _make_mock_response(
            "[mock] async openai client",
            model,
            4,
            2,
        )


class _MockChat:
    """Placeholder for .chat attribute."""

    def __init__(self, completions: Any) -> None:
        self.completions = completions


class MockOpenAISyncClient:
    """Full sync OpenAI client: .chat.completions.create (sync)."""

    def __init__(self) -> None:
        self.chat = _MockChat(_MockSyncCompletions())


class MockOpenAIAsyncClient:
    """Full async OpenAI client: .chat.completions.create (async)."""

    def __init__(self) -> None:
        self.chat = _MockChat(_MockAsyncCompletions())


# 2. Direct completions object: .create only, no .chat


class MockDirectCompletions:
    """Already a completions object (e.g. ChatOpenAI.client). Has .create,
    no .chat."""

    # pylint: disable=unused-argument
    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        await asyncio.sleep(0)
        return _make_mock_response(
            "[mock] direct completions",
            model,
            3,
            2,
        )


# 3. LangChain-like wrapper: .client (sync) + .async_client (async)


class _MockLangChainSyncCompletions:
    """Sync completions for LangChain .client attribute."""

    # pylint: disable=unused-argument
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        return _make_mock_response(
            "[mock] langchain sync client",
            model,
            6,
            3,
        )


class _MockLangChainAsyncCompletions:
    """Async completions for LangChain .async_client attribute."""

    # pylint: disable=unused-argument
    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        await asyncio.sleep(0)
        return _make_mock_response(
            "[mock] langchain async client",
            model,
            5,
            2,
        )


class MockLangChainLLM:
    """LangChain-like LLM wrapper exposing .client and .async_client."""

    def __init__(self) -> None:
        self.client = _MockLangChainSyncCompletions()
        self.async_client = _MockLangChainAsyncCompletions()


# 4. DashScope Generation-like: classmethod call


class MockDashScopeGeneration:
    """DashScope-like class with classmethod `call` for tracing route 4."""

    @classmethod
    def call(
        cls,
        model: Any = None,
        _prompt: Any = None,
        _history: Any = None,
        _api_key: Any = None,
        _messages: Any = None,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        # Match DashScope Generation.call signature: model can be positional
        # or keyword
        return _make_mock_response(
            "[mock] dashscope generation",
            model or "unknown",
            7,
            4,
        )


# ========================================================================== #
# Mock tools for trace_tool coverage                                         #
# ========================================================================== #


class MockBaseTool:
    """Minimal LangChain BaseTool duck-typing implementation.

    Used to test trace_tool across all input shapes:
    - single tool, list, tuple, Dict, ToolNode
    """

    def __init__(self, name: str, result: str = "ok") -> None:
        self.name = name
        self._result = result
        self.invoke_count = 0
        self.ainvoke_count = 0

    def invoke(
        self,
        input_data: Any,
        _config: Any = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        self.invoke_count += 1
        return {"tool": self.name, "result": self._result, "input": input_data}

    async def ainvoke(
        self,
        input_data: Any,
        _config: Any = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        self.ainvoke_count += 1
        await asyncio.sleep(0)
        return {"tool": self.name, "result": self._result, "input": input_data}


class MockMCPTool(MockBaseTool):
    """MCP-like tool with _mcp_tool marker.

    Used to test trace_tool auto-detection of MCP tools.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name, result="mcp_ok")
        self._mcp_tool = True  # MCP marker for auto-detection


class MockToolNode:
    """LangGraph ToolNode duck-typing implementation.

    Used to test trace_tool expansion of .tools_by_name.
    """

    def __init__(self, tools: List[Any]) -> None:
        self.tools_by_name = {t.name: t for t in tools}


class NotATool:
    """Unsupported object for trace_tool warning test."""


class DemoRolloutProcessor(AbstractRolloutProcessor):
    """
    Demo implementation of a Rollout Processor.
    Demonstrates how to construct a simple rollout execution flow (
    simulating Agent invocation).

    In production scenarios, this should invoke real Agent/model services,
    e.g., through OpenAI SDK or Dashscope SDK.
    """

    def setup(self) -> None:
        """
        Initialize workspace before processing requests.

        Creates mock LLM clients covering all four trace_client routing
        branches and instruments each one. This demonstrates that
        trace_client correctly detects client shape via duck typing and
        patches the appropriate method.
        """
        logger.info(
            "[DemoRolloutProcessor] setup() called - initializing workspace",
        )

        # Branch 1: Full OpenAI sync client (.chat.completions.create, sync)
        self._openai_sync = MockOpenAISyncClient()
        trace_client(self._openai_sync)
        logger.info("[DemoRolloutProcessor] MockOpenAISyncClient instrumented")

        # Branch 1: Full OpenAI async client (.chat.completions.create, async)
        self._openai_async = MockOpenAIAsyncClient()
        trace_client(self._openai_async)
        logger.info(
            "[DemoRolloutProcessor] MockOpenAIAsyncClient instrumented",
        )

        # Branch 2: Direct completions object (.create, no .chat)
        self._direct_completions = MockDirectCompletions()
        trace_client(self._direct_completions)
        logger.info(
            "[DemoRolloutProcessor] MockDirectCompletions instrumented",
        )

        # Branch 3: LangChain-like wrapper (.client / .async_client)
        self._langchain_llm = MockLangChainLLM()
        trace_client(self._langchain_llm)
        logger.info("[DemoRolloutProcessor] MockLangChainLLM instrumented")

        # Branch 4: DashScope Generation-like (classmethod call)
        trace_client(MockDashScopeGeneration)
        logger.info(
            "[DemoRolloutProcessor] MockDashScopeGeneration instrumented",
        )

        # ====================================================================
        # trace_tool coverage: all input shapes + MCP auto-detection
        # ====================================================================

        # 1. Single tool
        self._single_tool = MockBaseTool("single_tool")
        trace_tool(self._single_tool)
        logger.info("[DemoRolloutProcessor] single_tool instrumented")

        # 2. list[BaseTool]
        self._list_tools = [
            MockBaseTool("list_tool_0"),
            MockBaseTool("list_tool_1"),
        ]
        trace_tool(self._list_tools)
        logger.info("[DemoRolloutProcessor] list_tools instrumented")

        # 3. tuple[BaseTool]
        self._tuple_tools = (
            MockBaseTool("tuple_tool_0"),
            MockBaseTool("tuple_tool_1"),
        )
        trace_tool(self._tuple_tools)
        logger.info("[DemoRolloutProcessor] tuple_tools instrumented")

        # 4. Dict[str, BaseTool]
        self._dict_tools = {"dict_tool": MockBaseTool("dict_tool")}
        trace_tool(self._dict_tools)
        logger.info("[DemoRolloutProcessor] dict_tools instrumented")

        # 5. ToolNode (LangGraph)
        self._tool_node = MockToolNode(
            [
                MockBaseTool("node_tool_0"),
                MockBaseTool("node_tool_1"),
            ],
        )
        trace_tool(self._tool_node)
        logger.info("[DemoRolloutProcessor] tool_node instrumented")

        # 6. MCP tool (auto-detection, no explicit provider)
        self._mcp_tool = MockMCPTool("mcp_auto_tool")
        trace_tool(self._mcp_tool)  # should auto-set provider="mcp"
        logger.info(
            "[DemoRolloutProcessor] mcp_tool instrumented (auto-detected)",
        )

        # 7. Custom provider
        self._custom_tool = MockBaseTool("custom_provider_tool")
        trace_tool(self._custom_tool, provider="my-plugin")
        logger.info(
            "[DemoRolloutProcessor] custom_tool instrumented ("
            "provider=my-plugin)",
        )

        # 8. Unsupported object (triggers warning, no error)
        trace_tool(NotATool())
        logger.info(
            "[DemoRolloutProcessor] unsupported object handled (warning "
            "expected)",
        )

        # 9. Idempotency: patch same tool twice
        trace_tool(self._single_tool)  # should be no-op
        logger.info("[DemoRolloutProcessor] idempotency test passed")

        logger.info("[DemoRolloutProcessor] setup() completed")

    # ------------------------------------------------------------------
    # Internal helpers – each wrapped with a fine-grained observability
    # decorator so the Span tree looks like:
    #
    #   [ROLLOUT] ENTRY span              <- @observe_processor
    #     ├── [LLM]  _call_llm             <- @observe_llm
    #     │     ├── [LLM] openai_sync      <- trace_client branch 1
    #     │     ├── [LLM] openai_async     <- trace_client branch 1
    #     │     ├── [LLM] direct_comp      <- trace_client branch 2
    #     │     ├── [LLM] langchain_sync   <- trace_client branch 3
    #     │     ├── [LLM] langchain_async  <- trace_client branch 3
    #     │     └── [LLM] dashscope_gen    <- trace_client branch 4
    #     ├── [TOOL] _call_tools           <- @observe_tool
    #     │     ├── [TOOL] single_tool     <- trace_tool (single)
    #     │     ├── [TOOL] list_tool_*     <- trace_tool (list)
    #     │     ├── [TOOL] tuple_tool_*    <- trace_tool (tuple)
    #     │     ├── [TOOL] dict_tool       <- trace_tool (Dict)
    #     │     ├── [TOOL] node_tool_*     <- trace_tool (ToolNode)
    #     │     ├── [TOOL] mcp_auto_tool   <- trace_tool (MCP auto-detected)
    #     │     └── [TOOL] custom_tool     <- trace_tool (custom provider)
    #     └── [TOOL] _score_response       <- @observe_tool
    # ------------------------------------------------------------------

    @observe_llm
    async def _call_llm(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
    ) -> Any:
        """
        Simulated LLM call that exercises all trace_client routing branches.

        Each mock client is called to trigger its patched method, producing
        an LLM Span. The ``@observe_llm`` decorator wraps the whole function,
        so the outermost LLM Span is produced by the decorator itself.
        """
        # Branch 1: Full OpenAI sync client (sync call in async context)
        _ = self._openai_sync.chat.completions.create(
            model=model,
            messages=messages,
        )

        # Branch 1: Full OpenAI async client
        _ = await self._openai_async.chat.completions.create(
            model=model,
            messages=messages,
        )

        # Branch 2: Direct completions object
        _ = await self._direct_completions.create(
            model=model,
            messages=messages,
        )

        # Branch 3: LangChain-like wrapper (sync .client)
        _ = self._langchain_llm.client.create(model=model, messages=messages)

        # Branch 3: LangChain-like wrapper (async .async_client)
        _ = await self._langchain_llm.async_client.create(
            model=model,
            messages=messages,
        )

        # Branch 4: DashScope Generation-like (classmethod)
        _ = MockDashScopeGeneration.call(model=model, messages=messages)

        # Return a response for the @observe_llm wrapper to record
        return _make_mock_response(
            "[demo] aggregated response",
            model,
            30,
            16,
        )

    @observe_tool(name="trace_tool_tester")
    async def _call_tools(self) -> Dict[str, Any]:
        """Exercise all trace_tool-patched tools.

        Calls both sync invoke and async ainvoke on each tool shape
        to verify patching works correctly and spans are emitted.
        """
        results = {}

        # Single tool
        results["single_sync"] = self._single_tool.invoke("test_input")
        results["single_async"] = await self._single_tool.ainvoke("test_input")

        # List tools
        for tool in self._list_tools:
            results[f"list_{tool.name}_sync"] = tool.invoke("test_input")
            results[f"list_{tool.name}_async"] = await tool.ainvoke(
                "test_input",
            )

        # Tuple tools
        for tool in self._tuple_tools:
            results[f"tuple_{tool.name}_sync"] = tool.invoke("test_input")
            results[f"tuple_{tool.name}_async"] = await tool.ainvoke(
                "test_input",
            )

        # Dict tools
        for name, tool in self._dict_tools.items():
            results[f"dict_{name}_sync"] = tool.invoke("test_input")
            results[f"dict_{name}_async"] = await tool.ainvoke("test_input")

        # ToolNode tools
        for name, tool in self._tool_node.tools_by_name.items():
            results[f"node_{name}_sync"] = tool.invoke("test_input")
            results[f"node_{name}_async"] = await tool.ainvoke("test_input")

        # MCP tool (auto-detected provider)
        results["mcp_sync"] = self._mcp_tool.invoke("test_input")
        results["mcp_async"] = await self._mcp_tool.ainvoke("test_input")

        # Custom provider tool
        results["custom_sync"] = self._custom_tool.invoke("test_input")
        results["custom_async"] = await self._custom_tool.ainvoke("test_input")

        return results

    # pylint: disable=unused-argument
    @observe_tool(name="response_scorer")
    def _score_response(self, *, messages: List[Dict[str, str]]) -> float:
        """
        Simulated tool: assigns a fixed reward score to the LLM response.

        In production, this could call an external grader API or run a local
        reward model.  ``@observe_tool`` records the function name and
        arguments
        as a Tool Span.
        """
        # Demo: always return a fixed score
        return 0.95

    @observe_processor
    async def process(self, input_data: RolloutInput) -> RolloutOutput:
        """
        Main rollout entrypoint.

        Orchestrates the simulated Agent loop:
          1. ``_call_llm``      – invoke the language model  (LLM Span)
          2. ``_call_tools``    – exercise all trace_tool coverage (Tool Spans)
          3. ``_score_response`` – score the response        (Tool Span)

        Args:
            input_data: RolloutInput input parameter.

        Returns:
            RolloutOutput object containing execution results.
        """
        rollout_id = "unknown"
        if (
            input_data.request_metadata
            and input_data.request_metadata.rollout_id
        ):
            rollout_id = input_data.request_metadata.rollout_id
        elif (
            input_data.sample_extra and "rollout_id" in input_data.sample_extra
        ):
            rollout_id = input_data.sample_extra["rollout_id"]

        logger.info(
            f"[DemoRolloutProcessor] starting rollout | "
            f"model={input_data.model_resource.model_name}, "
            f"rollout_id={rollout_id}",
        )

        start = time.time()

        # Step 1: LLM call (produces an LLM Span via @observe_llm)
        llm_response = await self._call_llm(
            messages=input_data.messages or [],
            model=input_data.model_resource.model_name,
        )
        llm_content = (
            llm_response.choices[0].message.content
            if llm_response.choices
            else ""
        )

        # Step 2: Exercise trace_tool coverage (produces Tool Spans)
        tool_results = await self._call_tools()
        logger.info(
            f"[DemoRolloutProcessor] trace_tool results: "
            f"{len(tool_results)} calls",
        )

        # Step 3: Score the response (produces a Tool Span via @observe_tool)
        score = self._score_response(messages=input_data.messages or [])

        latency = round(time.time() - start, 4)

        agent_output = AgentOutput(
            messages=input_data.messages,
            sample_extra=input_data.rollout_extra,
            custom_metrics={"latency": latency, "llm_content": llm_content},
            reward_score=score,
        )

        result = RolloutOutput(
            agent_output=agent_output,
            status=TaskStatus.SUCCESS,
            error=None,
        )

        logger.info(
            f"[DemoRolloutProcessor][Async] result: rollout_id={rollout_id}, "
            f"latency={latency}s, score={score}",
        )
        return result
