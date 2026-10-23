# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
from typing import Any, AsyncGenerator, Dict, Generator, List, Union

from dashscope.legacy_1x.api_entities.dashscope_response import (
    MultiModalConversationResponse,
)
from dashscope.legacy_1x.client.base_api import BaseAioApi, BaseApi
from dashscope.legacy_1x.common.error import ModelRequired
from dashscope.legacy_1x.common.utils import _get_task_group_and_task
from dashscope.legacy_1x.utils.oss_utils import preprocess_message_element
from dashscope.legacy_1x.utils.param_utils import ParamUtil
from dashscope.legacy_1x.utils.message_utils import (
    merge_multimodal_single_response,
)


class MultiModalConversation(BaseApi):
    """MultiModal conversational robot interface."""

    task = "multimodal-generation"
    function = "generation"

    class Models:
        qwen_vl_chat_v1 = "qwen-vl-chat-v1"

    @classmethod
    # type: ignore
    def call(  # pylint: disable=arguments-renamed,too-many-branches,too-many-statements  # noqa: E501
        cls,
        model: str,
        messages: List = None,
        api_key: str = None,
        workspace: str = None,
        text: str = None,
        voice: str = None,
        language_type: str = None,
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
        n: int = None,
        ocr_options: Dict[str, Any] = None,
        logprobs: bool = None,
        top_logprobs: int = None,
        **kwargs,
    ) -> Union[
        MultiModalConversationResponse,
        Generator[
            MultiModalConversationResponse,
            None,
            None,
        ],
    ]:
        """Call the conversation model service.

        Args:
            model (str): The requested model, such as 'qwen-vl-max'.
            messages (list): The generation messages.
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.
            text (str): The text to generate.
            voice (str): The voice name for qwen tts.
            language_type (str): The synthesized language type.
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
            n (int, optional): Number of responses to generate (1-4).
            ocr_options (dict, optional): OCR task options for qwen-ocr models.
            logprobs (bool, optional): Whether to return log probabilities of
                the output tokens.
            top_logprobs (int, optional): Number of most likely tokens to
                return at each token position.
            **kwargs: Additional parameters passed to the API.

        Returns:
            Union[MultiModalConversationResponse,
                  Generator[MultiModalConversationResponse, None, None]]: If
            stream is True, return Generator, otherwise
        """
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
        if n is not None:
            kwargs["n"] = n
        if ocr_options is not None:
            kwargs["ocr_options"] = ocr_options
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        if model is None or not model:
            raise ModelRequired("Model is required!")
        task_group, _ = _get_task_group_and_task(__name__)
        input = {}  # pylint: disable=redefined-builtin
        msg_copy = None

        if messages is not None and messages:
            msg_copy = copy.deepcopy(messages)
            has_upload = cls._preprocess_messages(model, msg_copy, api_key)
            if has_upload:
                headers = kwargs.pop("headers", {})
                headers["X-DashScope-OssResourceResolve"] = "enable"
                kwargs["headers"] = headers

        if text is not None and text:
            input.update({"text": text})
        if voice is not None and voice:
            input.update({"voice": voice})
        if language_type is not None and language_type:
            input.update({"language_type": language_type})
        if msg_copy is not None:
            input.update({"messages": msg_copy})  # type: ignore

        # Validate input is not empty before sending request
        if not input:
            raise ValueError(
                "Input data is required. Please provide at least one of: "
                "messages, text, voice, or language_type.",
            )

        # Check if we need to merge incremental output
        is_incremental_output = kwargs.get("incremental_output", None)
        to_merge_incremental_output = False
        is_stream = kwargs.get("stream", False)
        if (
            ParamUtil.should_modify_incremental_output(model)
            and is_stream
            and is_incremental_output is not None
            and is_incremental_output is False
        ):
            to_merge_incremental_output = True
            kwargs["incremental_output"] = True

        # Pass incremental_to_full flag via user_agent parameter
        flag = "1" if to_merge_incremental_output else "0"
        existing_ua = kwargs.get("user_agent", "")
        new_ua = f"incremental_to_full/{flag}"
        kwargs["user_agent"] = (
            f"{existing_ua}; {new_ua}".strip() if existing_ua else new_ua
        )

        response = super().call(
            model=model,
            task_group=task_group,
            task=MultiModalConversation.task,
            function=MultiModalConversation.function,
            api_key=api_key,
            input=input,
            workspace=workspace,
            **kwargs,
        )
        if is_stream:
            if to_merge_incremental_output:
                # Extract n parameter for merge logic
                n = kwargs.get("n", 1)
                return cls._merge_multimodal_response(response, n)
            else:
                return (
                    MultiModalConversationResponse.from_api_response(rsp)
                    for rsp in response
                )
        else:
            return MultiModalConversationResponse.from_api_response(response)

    @classmethod
    def _preprocess_messages(
        cls,
        model: str,
        messages: List[dict],
        api_key: str,
    ):
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": ""},
                    {"text": ""},
                ]
            }
        ]
        """
        has_upload = False
        upload_certificate = None

        for message in messages:
            if message.get("role", "") == "user":
                content = message["content"]
                for elem in content:
                    if not isinstance(
                        elem,
                        (int, float, bool, str, bytes, bytearray),
                    ):
                        (
                            is_upload,
                            upload_certificate,
                        ) = preprocess_message_element(
                            model,
                            elem,
                            api_key,
                            upload_certificate,  # type: ignore[arg-type]
                        )
                        if is_upload and not has_upload:
                            has_upload = True
        return has_upload

    @classmethod
    def _merge_multimodal_response(
        cls,
        response,
        n=1,
    ) -> Generator[MultiModalConversationResponse, None, None]:
        """Merge incremental response chunks to simulate non-incremental output."""  # noqa: E501
        accumulated_data = {}

        for rsp in response:
            parsed_response = MultiModalConversationResponse.from_api_response(
                rsp,
            )
            result = merge_multimodal_single_response(
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


class AioMultiModalConversation(BaseAioApi):
    """Async MultiModal conversational robot interface."""

    task = "multimodal-generation"
    function = "generation"

    class Models:
        qwen_vl_chat_v1 = "qwen-vl-chat-v1"

    @classmethod  # type: ignore
    async def call(  # pylint: disable=arguments-renamed,too-many-branches,too-many-statements  # noqa: E501
        cls,
        model: str,
        messages: List = None,
        api_key: str = None,
        workspace: str = None,
        text: str = None,
        voice: str = None,
        language_type: str = None,
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
        n: int = None,
        ocr_options: Dict[str, Any] = None,
        logprobs: bool = None,
        top_logprobs: int = None,
        **kwargs,
    ) -> Union[
        MultiModalConversationResponse,
        AsyncGenerator[
            MultiModalConversationResponse,
            None,
        ],
    ]:
        """Call the conversation model service asynchronously.

        Args:
            model (str): The requested model, such as 'qwen-vl-max'.
            messages (list): The generation messages.
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.
            text (str): The text to generate.
            voice (str): The voice name for qwen tts.
            language_type (str): The synthesized language type.
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
            n (int, optional): Number of responses to generate (1-4).
            ocr_options (dict, optional): OCR task options for qwen-ocr models.
            logprobs (bool, optional): Whether to return log probabilities of
                the output tokens.
            top_logprobs (int, optional): Number of most likely tokens to
                return at each token position.
            **kwargs: Additional parameters passed to the API.

        Returns:
            Union[MultiModalConversationResponse,
                  AsyncGenerator[MultiModalConversationResponse, None]]: If
            stream is True, return AsyncGenerator, otherwise
        """
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
        if n is not None:
            kwargs["n"] = n
        if ocr_options is not None:
            kwargs["ocr_options"] = ocr_options
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        if model is None or not model:
            raise ModelRequired("Model is required!")
        task_group, _ = _get_task_group_and_task(__name__)
        input = {}  # pylint: disable=redefined-builtin
        msg_copy = None

        if messages is not None and messages:
            msg_copy = copy.deepcopy(messages)
            has_upload = cls._preprocess_messages(model, msg_copy, api_key)
            if has_upload:
                headers = kwargs.pop("headers", {})
                headers["X-DashScope-OssResourceResolve"] = "enable"
                kwargs["headers"] = headers

        if text is not None and text:
            input.update({"text": text})
        if voice is not None and voice:
            input.update({"voice": voice})
        if language_type is not None and language_type:
            input.update({"language_type": language_type})
        if msg_copy is not None:
            input.update({"messages": msg_copy})  # type: ignore

        # Validate input is not empty before sending request
        if not input:
            raise ValueError(
                "Input data is required. Please provide at least one of: "
                "messages, text, voice, or language_type.",
            )
        # Check if we need to merge incremental output
        is_incremental_output = kwargs.get("incremental_output", None)
        to_merge_incremental_output = False
        is_stream = kwargs.get("stream", False)
        if (
            ParamUtil.should_modify_incremental_output(model)
            and is_stream
            and is_incremental_output is not None
            and is_incremental_output is False
        ):
            to_merge_incremental_output = True
            kwargs["incremental_output"] = True

        # Pass incremental_to_full flag via headers user-agent
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        flag = "1" if to_merge_incremental_output else "0"
        kwargs["headers"]["user-agent"] = (
            kwargs["headers"].get("user-agent", "")
            + f"; incremental_to_full/{flag}"
        )

        response = await super().call(
            model=model,
            task_group=task_group,
            task=AioMultiModalConversation.task,
            function=AioMultiModalConversation.function,
            api_key=api_key,
            input=input,
            workspace=workspace,
            **kwargs,
        )
        if is_stream:
            if to_merge_incremental_output:
                # Extract n parameter for merge logic
                n = kwargs.get("n", 1)
                return cls._merge_multimodal_response(response, n)
            else:
                return cls._stream_responses(response)
        else:
            return MultiModalConversationResponse.from_api_response(response)

    @classmethod
    def _preprocess_messages(
        cls,
        model: str,
        messages: List[dict],
        api_key: str,
    ):
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": ""},
                    {"text": ""},
                ]
            }
        ]
        """
        has_upload = False
        upload_certificate = None

        for message in messages:
            if message.get("role", "") == "user":
                content = message["content"]
                for elem in content:
                    if not isinstance(
                        elem,
                        (int, float, bool, str, bytes, bytearray),
                    ):
                        (
                            is_upload,
                            upload_certificate,
                        ) = preprocess_message_element(
                            model,
                            elem,
                            api_key,
                            upload_certificate,  # type: ignore[arg-type]
                        )
                        if is_upload and not has_upload:
                            has_upload = True
        return has_upload

    @classmethod
    async def _stream_responses(
        cls,
        response,
    ) -> AsyncGenerator[MultiModalConversationResponse, None]:
        """Convert async response stream to MultiModalConversationResponse stream."""  # noqa: E501
        # Type hint: when stream=True, response is actually an AsyncIterable
        async for rsp in response:  # type: ignore
            yield MultiModalConversationResponse.from_api_response(rsp)

    @classmethod
    async def _merge_multimodal_response(
        cls,
        response,
        n=1,
    ) -> AsyncGenerator[MultiModalConversationResponse, None]:
        """Async version of merge incremental response chunks."""
        accumulated_data = {}

        async for rsp in response:
            parsed_response = MultiModalConversationResponse.from_api_response(
                rsp,
            )
            result = merge_multimodal_single_response(
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
