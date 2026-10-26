# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import asyncio
import json
import logging
import math
import multiprocessing
import operator
import socket
import time
from typing import Dict, List
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Public names are provided lazily via __getattr__; __all__ entries are not
# bindings.
# pylint: disable=undefined-all-variable
from dashscope.finetune.reinforcement import RolloutInput, RolloutOutput
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    AgentOutput,
    TaskStatus,
)

# pylint: disable=no-name-in-module
from dashscope.finetune.reinforcement.component.observability import (
    observe_processor,
    trace_client,
    trace_tool,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRolloutProcessor,
)

logger = logging.getLogger(__name__)

MCP_PORT = 10086


# ========================================================================== #
#                             MCP SERVER                                     #
# ========================================================================== #


def _evaluate_exp(expression: str) -> str:
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    allowed_names = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("__")
    }
    allowed_names.update({"pi": math.pi, "e": math.e})

    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Unknown identifier: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = eval_expr(node.left)
            right = eval_expr(node.right)
            if type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -eval_expr(node.operand)
        elif isinstance(node, ast.Call):
            func = eval_expr(node.func)
            args = [eval_expr(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported operation: {ast.dump(node)}")

    expression = (
        expression.replace("^", "**").replace("ร�", "*").replace("รท", "/")
    )
    parsed_expr = ast.parse(expression, mode="eval")
    evaluation_result = eval_expr(parsed_expr.body)
    return str(evaluation_result)


mcp = FastMCP("calculator", port=MCP_PORT)


@mcp.tool()
async def calculate(expression: str) -> str:
    """Calculates/evaluates the given expression."""
    return _evaluate_exp(expression)


def _run_mcp_server():
    mcp.run(transport="sse")


def _wait_for_port(
    port: int,
    host: str = "localhost",
    timeout: float = 30.0,
) -> bool:
    start_time = time.perf_counter()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError as exc:
            time.sleep(0.5)
            if time.perf_counter() - start_time >= timeout:
                raise TimeoutError(
                    f"[DashSystem] Cannot connect to {port} in {timeout} s!",
                ) from exc


# ========================================================================== #
#                          ROLLOUT PROCESSOR                                 #
# ========================================================================== #


class CalcXRolloutProcessor(AbstractRolloutProcessor):
    """
    Rollout Processor for Calc-X using MCP Tools.
    Manages a persistent MCP Server process and shared LangGraph agent.
    """

    def _is_port_in_use(self, port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0

    def __init__(self):
        super().__init__()
        self._init_lock = asyncio.Lock()

        if self._is_port_in_use(MCP_PORT):
            logger.info(
                f"MCP Server already running on port {MCP_PORT}. Skipping "
                f"initialization.",
            )
            self.mcp_process = None
        else:
            logger.info(
                f"Port {MCP_PORT} is free. Starting MCP Server process...",
            )
            self.mcp_process = multiprocessing.Process(
                target=_run_mcp_server,
                daemon=True,
            )
            self.mcp_process.start()
            try:
                _wait_for_port(MCP_PORT)
            except TimeoutError as e:
                logger.error(f"Failed to start MCP Server: {e}")
                raise

        self._shared_mcp_client = None
        self._shared_tools = None
        self._shared_graph = None

    async def _async_setup(self):
        if not self._shared_mcp_client:
            start_time = time.perf_counter()
            async with self._init_lock:
                if not self._shared_mcp_client:
                    await self._init_resources_async()
                    elapsed = time.perf_counter() - start_time
                    logger.info(f"[DEBUG] _async_setup: {elapsed:.4f}s")

    def _get_mcp_servers(self) -> Dict:
        return {
            "calculator": {
                "transport": "sse",
                "url": f"http://localhost:{MCP_PORT}/sse",
            },
        }

    async def _init_resources_async(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ):
        logger.info(
            "Initializing shared MCP Client and Graph for this worker "
            "instance...",
        )
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                client = MultiServerMCPClient(self._get_mcp_servers())
                tools = await client.get_tools()
                trace_tool(tools)  # Enable tool tracing for MCP tools
                graph = self._build_graph(tools)

                self._shared_tools = tools
                self._shared_graph = graph
                self._shared_mcp_client = client
                logger.info(
                    "Shared MCP Client successfully initialized. Tools "
                    "cached.",
                )
                return
            except Exception as e:
                last_exc = e
                delay = base_delay * attempt
                logger.warning(
                    f"MCP Client init attempt {attempt}/{max_retries} "
                    f"failed: {e}. "
                    f"Retrying in {delay}s...",
                )
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"Failed to initialize MCP Client after {max_retries} attempts",
        ) from last_exc

    # ------------------------------------------------------------------ #
    #  LangGraph construction                                             #
    # ------------------------------------------------------------------ #

    async def _call_model(self, state: MessagesState, config: RunnableConfig):
        model = config["configurable"]["model"]
        try:
            response = await model.ainvoke(state["messages"], config=config)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Model invoke error with {model.model}: {e}")
            return {
                "messages": [("ai", f"Error: Model invocation failed. {e}")],
            }

    def _should_continue(self, state: MessagesState):
        last_message = state["messages"][-1]
        if (
            hasattr(last_message, "tool_calls")
            and len(last_message.tool_calls) > 0
        ):
            return "tools"
        return END

    def _build_graph(self, tools: List[BaseTool]) -> StateGraph:
        workflow = StateGraph(MessagesState)
        standard_tool_node = ToolNode(tools)

        async def tool_node_with_logging(
            state: MessagesState,
            config: RunnableConfig,
        ):
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for call in last_message.tool_calls:
                    logger.info(
                        f"[Tool Call] Tool: {call.get('name')}, Args:"
                        f" {call.get('args')}",
                    )
            return await standard_tool_node.ainvoke(state, config)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node_with_logging)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #

    def _build_llm(self, input_data: RolloutInput) -> ChatOpenAI:
        resource = input_data.model_resource
        params = input_data.sampling_params or {}
        api_key = (
            resource.api_key.get_secret_value()
            if hasattr(resource.api_key, "get_secret_value")
            else resource.api_key
        )
        extra_kwargs = {
            k: v
            for k, v in params.items()
            if k not in ("temperature", "max_tokens", "timeout", "max_turns")
        }
        llm = ChatOpenAI(
            model=resource.model_name,
            openai_api_key=api_key,
            openai_api_base=resource.base_url,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 2048),
            request_timeout=params.get("timeout", 60.0),
            model_kwargs=extra_kwargs,
            streaming=False,
        )
        # Enable LLM tracing for this ChatOpenAI instance.
        # Works transparently for LangChain because trace_client detects
        # the .client / .async_client structure via duck typing.
        trace_client(llm)
        return llm

    @staticmethod
    def _to_langchain_messages(messages: List[Dict]) -> List:
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(ChatMessage(role=role, content=content))
        return lc_messages

    # ------------------------------------------------------------------ #
    #  Core rollout logic                                                 #
    # ------------------------------------------------------------------ #

    async def _async_process(self, input_data: RolloutInput) -> RolloutOutput:
        rollout_id = "unknown"
        if (
            input_data.request_metadata
            and input_data.request_metadata.rollout_id
        ):
            rollout_id = input_data.request_metadata.rollout_id
        elif (
            input_data.rollout_extra
            and "rollout_id" in input_data.rollout_extra
        ):
            rollout_id = input_data.rollout_extra["rollout_id"]

        logger.info(
            f"[CalcXRolloutProcessor] Starting rollout {rollout_id} "
            f"with model {input_data.model_resource.model_name} at"
            f" {input_data.model_resource.base_url}",
        )

        start_time = time.time()

        try:
            tools = self._shared_tools
            graph = self._shared_graph

            llm = self._build_llm(input_data)
            model_with_tools = llm.bind_tools(tools)

            messages = self._to_langchain_messages(input_data.messages)

            max_turns = (input_data.sampling_params or {}).get("max_turns", 10)
            config = RunnableConfig(
                recursion_limit=max_turns * 2 + 5,
                configurable={"model": model_with_tools},
                metadata={
                    "rollout_id": rollout_id,
                    "model_name": input_data.model_resource.model_name,
                    "base_url": input_data.model_resource.base_url,
                },
            )

            final_state = await graph.ainvoke(
                {"messages": messages},
                config=config,
            )
            final_messages = final_state["messages"]
            output_content = str(final_messages[-1].content)

            reward = 0.0

            latency = round(time.time() - start_time, 4)
            agent_output = AgentOutput(
                messages=convert_to_openai_messages(final_messages),
                rollout_extra=input_data.rollout_extra,
                rollout_metrics={"latency": latency},
                reward_score=reward,
            )

            logger.info(
                f"Rollout result {rollout_id} | latency: {latency}s | "
                f"reward: {reward} | content: {json.dumps(output_content)}",
            )
            return RolloutOutput(
                agent_output=agent_output,
                status=TaskStatus.SUCCESS,
                error=None,
            )

        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            latency = round(time.time() - start_time, 4)

            return RolloutOutput(
                agent_output=AgentOutput(
                    messages=[{"role": "system", "content": f"Error: {e}"}],
                    rollout_extra=input_data.rollout_extra,
                    rollout_metrics={"latency": latency},
                    reward_score=0.0,
                ),
                status=TaskStatus.FAILED,
                error=str(e),
            )

    @observe_processor
    async def process(self, input_data: RolloutInput) -> RolloutOutput:
        await self._async_setup()
        return await self._async_process(input_data)

    def __del__(self):
        if (
            hasattr(self, "mcp_process")
            and self.mcp_process is not None
            and self.mcp_process.is_alive()
        ):
            try:
                self.mcp_process.terminate()
                self.mcp_process.join(timeout=1)
                logger.info("MCP Server process terminated.")
            except Exception as e:
                logger.warning(f"Error terminating MCP Server process: {e}")


if __name__ == "__main__":
    from dashscope.finetune.reinforcement.component.data.base_data_model import (  # noqa: E501
        ModelResource,
    )
    import os

    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

    input_format = (
        "Output the answer when you are ready. The answer should be "
        "surrounded by three sharps (`###`), in the form of ### ANSWER: "
        "<answer> ###."
    )

    rollout_input = RolloutInput(
        messages=[
            {
                "role": "user",
                "content": "6.6 minus x (3/2) times equals 5.6."
                + " "
                + input_format,
            },
        ],
        ground_truth="2/3",
        model_resource=ModelResource(
            model_name="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-XXXX",
        ),
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 2048,
            "max_turns": 10,
        },
    )
    print(json.dumps(rollout_input.model_dump(mode="json"), indent=4))

    processor = CalcXRolloutProcessor()
    result = asyncio.run(processor.process(rollout_input))
    print(result)
