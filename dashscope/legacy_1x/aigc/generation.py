# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import json
from typing import Any, Dict, Generator, List, Union, AsyncGenerator

from dashscope.legacy_1x.api_entities.dashscope_response import (
    GenerationResponse,
    Message,
    Role,
)
from dashscope.legacy_1x.client.base_api import BaseAioApi, BaseApi
from dashscope.legacy_1x.common.constants import (
    CUSTOMIZED_MODEL_ID,
    DEPRECATED_MESSAGE,
    HISTORY,
    MESSAGES,
    PROMPT,
)
from dashscope.legacy_1x.common.error import InputRequired, ModelRequired
from dashscope.legacy_1x.common.logging import logger
from dashscope.legacy_1x.common.utils import _get_task_group_and_task
from dashscope.legacy_1x.utils.param_utils import ParamUtil
from dashscope.legacy_1x.utils.message_utils import merge_single_response


class Generation(BaseApi):
    task = "text-generation"
    """API for AI-Generated Content(AIGC) models.

    """

    class Models:
        """@deprecated, use qwen_turbo instead"""

        qwen_v1 = "qwen-v1"
        """@deprecated, use qwen_plus instead"""
        qwen_plus_v1 = "qwen-plus-v1"

        bailian_v1 = "bailian-v1"
        dolly_12b_v2 = "dolly-12b-v2"
        qwen_turbo = "qwen-turbo"
        qwen_plus = "qwen-plus"
        qwen_max = "qwen-max"

    @classmethod
    # type: ignore[override]
    def call(  # pylint: disable=arguments-renamed,too-many-branches,too-many-statements  # type: ignore[override]  # noqa: E501
        cls,
        model: str,
        prompt: Any = None,
        history: list = None,
        api_key: str = None,
        messages: List[Message] = None,
        plugins: Union[str, Dict[str, Any]] = None,
        workspace: str = None,
        stream: bool = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        max_tokens: int = None,
        seed: int = None,
        stop: Union[str, List] = None,
        repetition_penalty: float = None,
        presence_penalty: float = None,
        result_format: str = None,
        incremental_output: bool = None,
        enable_search: bool = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: Union[str, Dict[str, Any]] = None,
        enable_thinking: bool = None,
        thinking_budget: int = None,
        n: int = None,
        logprobs: bool = None,
        top_logprobs: int = None,
        search_options: Dict[str, Any] = None,
        parallel_tool_calls: bool = None,
        response_format: Dict[str, Any] = None,
        output_format: str = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Union[GenerationResponse, Generator[GenerationResponse, None, None]]:
        """Call generation model service.

        Args:
            model (str): The requested model, such as qwen-turbo.
            prompt (Any): The input prompt.
            history (list): The user provided history, deprecated.
            api_key (str, optional): The api api_key, can be None.
            messages (list): The generation messages.
            plugins (Any): The plugin config, str or dict.
            workspace (str): The dashscope workspace id.
            stream (bool, optional): Enable streaming output.
            temperature (float, optional): Controls randomness, range [0, 2).
            top_p (float, optional): Nucleus sampling, range (0, 1.0].
            top_k (int, optional): Size of candidate token set for sampling.
            max_tokens (int, optional): Maximum output token count.
            seed (int, optional): Random seed for reproducibility.
            stop (str or list, optional): Stop sequences.
            repetition_penalty (float, optional): Penalizes repeated sequences.
                1.0 means no penalty.
            presence_penalty (float, optional): Controls content repetition,
                range [-2.0, 2.0].
            result_format (str, optional): "message" or "text".
            incremental_output (bool, optional): In streaming mode, output only
                new tokens (True) vs. cumulative output (False).
            enable_search (bool, optional): Enable web search.
            tools (list, optional): Tool definitions for function calling.
            tool_choice (str or dict, optional): Tool selection strategy.
            enable_thinking (bool, optional): Enable thinking mode for
                hybrid thinking models.
            thinking_budget (int, optional): Maximum token budget for thinking
                mode. Controls the maximum length of thinking process.
            n (int, optional): Number of responses to generate (1-4).
            logprobs (bool, optional): Whether to return log probabilities of
                the output tokens.
            top_logprobs (int, optional): Number of most likely tokens to
                return at each token position when logprobs is enabled.
            search_options (dict, optional): Configuration options for web
                search feature.
            parallel_tool_calls (bool, optional): Enable parallel tool calls
                for function calling.
            response_format (dict, optional): Format constraint for response,
                e.g., {"type": "json_object"} for JSON mode.
            output_format (str, optional): Output format for qwen-deep-research
                model. Options: "model_detailed_report" (default),
                "model_summary_report".
            **kwargs: Additional parameters passed to the API.

        Returns:
            Union[GenerationResponse,
                  Generator[GenerationResponse, None, None]]: If
            stream is True, return Generator, otherwise GenerationResponse.
        """
        if (prompt is None or not prompt) and (
            messages is None or not messages
        ):
            raise InputRequired("prompt or messages is required!")
        if model is None or not model:
            raise ModelRequired("Model is required!")
        if stream is not None:
            kwargs["stream"] = stream
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if stop is not None:
            kwargs["stop"] = stop
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if result_format is not None:
            kwargs["result_format"] = result_format
        if incremental_output is not None:
            kwargs["incremental_output"] = incremental_output
        if enable_search is not None:
            kwargs["enable_search"] = enable_search
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        if n is not None:
            kwargs["n"] = n
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        if search_options is not None:
            kwargs["search_options"] = search_options
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        if response_format is not None:
            kwargs["response_format"] = response_format
        task_group, function = _get_task_group_and_task(__name__)
        if plugins is not None:
            headers = kwargs.pop("headers", {})
            if isinstance(plugins, str):
                headers["X-DashScope-Plugin"] = plugins
            else:
                headers["X-DashScope-Plugin"] = json.dumps(plugins)
            kwargs["headers"] = headers
        (
            input,  # pylint: disable=redefined-builtin
            parameters,
        ) = cls._build_input_parameters(
            model,
            prompt,
            history,
            messages,
            **kwargs,
        )

        is_stream = parameters.get("stream", False)
        # Check if we need to merge incremental output
        is_incremental_output = kwargs.get("incremental_output", None)
        to_merge_incremental_output = False
        if (
            ParamUtil.should_modify_incremental_output(model)
            and is_stream
            and is_incremental_output is False
        ):
            to_merge_incremental_output = True
            parameters["incremental_output"] = True

        # Pass incremental_to_full flag via user_agent parameter
        flag = "1" if to_merge_incremental_output else "0"
        existing_ua = parameters.get("user_agent", "")
        new_ua = f"incremental_to_full/{flag}"
        parameters["user_agent"] = (
            f"{existing_ua}; {new_ua}".strip() if existing_ua else new_ua
        )

        response = super().call(
            model=model,
            task_group=task_group,
            task=Generation.task,
            function=function,
            api_key=api_key,
            input=input,
            workspace=workspace,
            **parameters,
        )
        if is_stream:
            if to_merge_incremental_output:
                # Extract n parameter for merge logic
                n = parameters.get("n", 1)
                return cls._merge_generation_response(response, n)
            else:
                return (
                    GenerationResponse.from_api_response(rsp)
                    for rsp in response
                )
        else:
            return GenerationResponse.from_api_response(response)

    @classmethod
    def _build_input_parameters(
        cls,
        model,
        prompt,
        history,
        messages,
        **kwargs,
    ):
        if model == Generation.Models.qwen_v1:
            logger.warning(
                "Model %s is deprecated, use %s instead!",
                Generation.Models.qwen_v1,
                Generation.Models.qwen_turbo,
            )
        if model == Generation.Models.qwen_plus_v1:
            logger.warning(
                "Model %s is deprecated, use %s instead!",
                Generation.Models.qwen_plus_v1,
                Generation.Models.qwen_plus,
            )
        parameters = {}
        input = {}  # pylint: disable=redefined-builtin
        if history is not None:
            logger.warning(DEPRECATED_MESSAGE)
            input[HISTORY] = history
            if prompt is not None and prompt:
                input[PROMPT] = prompt
        elif messages is not None:
            msgs = copy.deepcopy(messages)
            if prompt is not None and prompt:
                msgs.append({"role": Role.USER, "content": prompt})
            input = {MESSAGES: msgs}
        else:
            input[PROMPT] = prompt

        if model.startswith("qwen"):
            enable_search = kwargs.pop("enable_search", False)
            if enable_search:
                parameters["enable_search"] = enable_search
        elif model.startswith("bailian"):
            customized_model_id = kwargs.pop("customized_model_id", None)
            if customized_model_id is None:
                raise InputRequired(
                    f"customized_model_id is required for {model}",
                )
            input[CUSTOMIZED_MODEL_ID] = customized_model_id

        return input, {**parameters, **kwargs}

    @classmethod
    def _merge_generation_response(
        cls,
        response,
        n=1,
    ) -> Generator[GenerationResponse, None, None]:
        """Merge incremental response chunks to simulate non-incremental output."""  # noqa: E501
        accumulated_data = {}
        for rsp in response:
            parsed_response = GenerationResponse.from_api_response(rsp)
            result = merge_single_response(
                parsed_response,
                accumulated_data,
                n,
            )
            if result is True:
                yield parsed_response
            elif isinstance(result, list):
                # Multiple responses to yield (for n>1 non-stop cases)
                for resp in result:
                    yield resp


class AioGeneration(BaseAioApi):
    task = "text-generation"
    """API for AI-Generated Content(AIGC) models.

    """

    class Models:
        """@deprecated, use qwen_turbo instead"""

        qwen_v1 = "qwen-v1"
        """@deprecated, use qwen_plus instead"""
        qwen_plus_v1 = "qwen-plus-v1"

        bailian_v1 = "bailian-v1"
        dolly_12b_v2 = "dolly-12b-v2"
        qwen_turbo = "qwen-turbo"
        qwen_plus = "qwen-plus"
        qwen_max = "qwen-max"

    # type: ignore[override]
    @classmethod
    async def call(  # type: ignore[override] # pylint: disable=arguments-renamed,too-many-branches,too-many-statements # noqa: E501
        # type: ignore[override]
        cls,
        model: str,
        prompt: Any = None,
        history: list = None,
        api_key: str = None,
        messages: List[Message] = None,
        plugins: Union[str, Dict[str, Any]] = None,
        workspace: str = None,
        stream: bool = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        max_tokens: int = None,
        seed: int = None,
        stop: Union[str, List] = None,
        repetition_penalty: float = None,
        presence_penalty: float = None,
        result_format: str = None,
        incremental_output: bool = None,
        enable_search: bool = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: Union[str, Dict[str, Any]] = None,
        enable_thinking: bool = None,
        thinking_budget: int = None,
        n: int = None,
        logprobs: bool = None,
        top_logprobs: int = None,
        search_options: Dict[str, Any] = None,
        parallel_tool_calls: bool = None,
        response_format: Dict[str, Any] = None,
        output_format: str = None,
        **kwargs,
    ) -> Union[GenerationResponse, AsyncGenerator[GenerationResponse, None]]:
        """Call generation model service.

        Args:
            model (str): The requested model, such as qwen-turbo.
            prompt (Any): The input prompt.
            history (list): The user provided history, deprecated.
            api_key (str, optional): The api api_key, can be None.
            messages (list): The generation messages.
            plugins (Any): The plugin config, str or dict.
            workspace (str): The dashscope workspace id.
            stream (bool, optional): Enable streaming output.
            temperature (float, optional): Controls randomness, range [0, 2).
            top_p (float, optional): Nucleus sampling, range (0, 1.0].
            top_k (int, optional): Size of candidate token set for sampling.
            max_tokens (int, optional): Maximum output token count.
            seed (int, optional): Random seed for reproducibility.
            stop (str or list, optional): Stop sequences.
            repetition_penalty (float, optional): Penalizes repeated sequences.
            presence_penalty (float, optional): Controls content repetition.
            result_format (str, optional): "message" or "text".
            incremental_output (bool, optional): In streaming mode, output only
                new tokens (True) vs. cumulative output (False).
            enable_search (bool, optional): Enable web search.
            tools (list, optional): Tool definitions for function calling.
            tool_choice (str or dict, optional): Tool selection strategy.
            enable_thinking (bool, optional): Enable thinking mode.
            thinking_budget (int, optional): Maximum token budget for thinking
                mode. Controls the maximum length of thinking process.
            n (int, optional): Number of responses to generate (1-4).
            logprobs (bool, optional): Whether to return log probabilities of
                the output tokens.
            top_logprobs (int, optional): Number of most likely tokens to
                return at each token position when logprobs is enabled.
            search_options (dict, optional): Configuration options for web
                search feature.
            parallel_tool_calls (bool, optional): Enable parallel tool calls
                for function calling.
            response_format (dict, optional): Format constraint for response,
                e.g., {"type": "json_object"} for JSON mode.
            output_format (str, optional): Output format for qwen-deep-research
                model. Options: "model_detailed_report" (default),
                "model_summary_report".
            **kwargs: Additional parameters passed to the API.

        Returns:
            Union[GenerationResponse,
                  AsyncGenerator[GenerationResponse, None]]: If
            stream is True, return AsyncGenerator, else GenerationResponse.
        """
        if (prompt is None or not prompt) and (
            messages is None or not messages
        ):
            raise InputRequired("prompt or messages is required!")
        if model is None or not model:
            raise ModelRequired("Model is required!")
        if stream is not None:
            kwargs["stream"] = stream
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if stop is not None:
            kwargs["stop"] = stop
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if result_format is not None:
            kwargs["result_format"] = result_format
        if incremental_output is not None:
            kwargs["incremental_output"] = incremental_output
        if enable_search is not None:
            kwargs["enable_search"] = enable_search
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        if n is not None:
            kwargs["n"] = n
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        if search_options is not None:
            kwargs["search_options"] = search_options
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        if response_format is not None:
            kwargs["response_format"] = response_format
        if output_format is not None:
            kwargs["output_format"] = output_format
        task_group, function = _get_task_group_and_task(__name__)
        if plugins is not None:
            headers = kwargs.pop("headers", {})
            if isinstance(plugins, str):
                headers["X-DashScope-Plugin"] = plugins
            else:
                headers["X-DashScope-Plugin"] = json.dumps(plugins)
            kwargs["headers"] = headers
        # pylint: disable=protected-access
        (
            input,  # pylint: disable=redefined-builtin
            parameters,
        ) = Generation._build_input_parameters(
            model,
            prompt,
            history,
            messages,
            **kwargs,
        )

        is_stream = parameters.get("stream", False)
        # Check if we need to merge incremental output
        is_incremental_output = kwargs.get("incremental_output", None)
        to_merge_incremental_output = False
        if (
            ParamUtil.should_modify_incremental_output(model)
            and is_stream
            and is_incremental_output is False
        ):
            to_merge_incremental_output = True
            parameters["incremental_output"] = True

        # Pass incremental_to_full flag via user_agent parameter
        flag = "1" if to_merge_incremental_output else "0"
        existing_ua = parameters.get("user_agent", "")
        new_ua = f"incremental_to_full/{flag}"
        parameters["user_agent"] = (
            f"{existing_ua}; {new_ua}".strip() if existing_ua else new_ua
        )

        response = await super().call(
            model=model,
            task_group=task_group,
            task=Generation.task,
            function=function,
            api_key=api_key,
            input=input,
            workspace=workspace,
            **parameters,
        )
        if is_stream:
            if to_merge_incremental_output:
                # Extract n parameter for merge logic
                n = parameters.get("n", 1)
                return cls._merge_generation_response(response, n)
            else:
                return cls._stream_responses(response)
        else:
            return GenerationResponse.from_api_response(response)

    @classmethod
    async def _stream_responses(
        cls,
        response,
    ) -> AsyncGenerator[GenerationResponse, None]:
        """Convert async response stream to GenerationResponse stream."""
        # Type hint: when stream=True, response is actually an AsyncIterable
        async for rsp in response:  # type: ignore
            yield GenerationResponse.from_api_response(rsp)

    @classmethod
    async def _merge_generation_response(
        cls,
        response,
        n=1,
    ) -> AsyncGenerator[GenerationResponse, None]:
        """Async version of merge incremental response chunks."""
        accumulated_data = {}

        async for rsp in response:  # type: ignore
            parsed_response = GenerationResponse.from_api_response(rsp)
            result = merge_single_response(
                parsed_response,
                accumulated_data,
                n,
            )
            if result is True:
                yield parsed_response
            elif isinstance(result, list):
                # Multiple responses to yield (for n>1 non-stop cases)
                for resp in result:
                    yield resp
