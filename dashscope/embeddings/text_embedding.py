# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List, Union

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.client.base_api import BaseApi
from dashscope.common.constants import TEXT_EMBEDDING_INPUT_KEY
from dashscope.common.utils import _get_task_group_and_task


class TextEmbedding(BaseApi):
    task = "text-embedding"

    class Models:
        text_embedding_v1 = "text-embedding-v1"
        text_embedding_v2 = "text-embedding-v2"
        text_embedding_v3 = "text-embedding-v3"
        text_embedding_v4 = "text-embedding-v4"
        qwen3_7_text_embedding = "qwen3.7-text-embedding"

    @classmethod
    def call(  # type: ignore[override]  # pylint: disable=arguments-renamed
        cls,
        model: str,
        input: Union[str, List[str]],  # pylint: disable=redefined-builtin
        workspace: str = None,
        api_key: str = None,
        text_type: str = None,
        dimension: int = None,
        output_type: str = None,
        instruct: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Get embedding of text input.

        Args:
            model (str): The embedding model name.
            input (Union[str, List[str], io.IOBase]): The text input,
                can be a text or list of text or opened file object,
                if opened file object, will read all lines,
                one embedding per line.
            workspace (str): The dashscope workspace id.
            text_type (str, optional): "query" for search queries,
                "document" (default) for corpus/symmetric tasks.
            dimension (int, optional): Output vector dimension.
                Options: 2048 (v4 only), 1536 (v4 only), 1024 (default),
                768, 512, 256, 128, 64. Only for v3/v4.
            output_type (str, optional): Output format: "dense" (default),
                "sparse", or "dense&sparse". Only for v3/v4.
            instruct (str, optional): Custom task instruction to guide
                model understanding of query intent.
            **kwargs:
                Additional parameters passed to the API.

        Returns:
            DashScopeAPIResponse: The embedding result.
        """
        embedding_input = {}
        if isinstance(input, str):
            embedding_input[TEXT_EMBEDDING_INPUT_KEY] = [input]
        else:
            embedding_input[TEXT_EMBEDDING_INPUT_KEY] = input
        kwargs.pop("stream", False)  # not support streaming output.
        if text_type is not None:
            kwargs["text_type"] = text_type
        if dimension is not None:
            kwargs["dimension"] = dimension
        if output_type is not None:
            kwargs["output_type"] = output_type
        if instruct is not None:
            kwargs["instruct"] = instruct
        task_group, function = _get_task_group_and_task(__name__)
        return super().call(
            model=model,
            input=embedding_input,
            task_group=task_group,
            task=TextEmbedding.task,
            function=function,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )
