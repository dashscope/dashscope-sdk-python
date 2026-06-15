# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple

from dashscope.api_entities.dashscope_response import ReRankResponse
from dashscope.client.base_api import BaseAioApi, BaseApi
from dashscope.common.error import InputRequired, ModelRequired
from dashscope.common.utils import _get_task_group_and_task

__all__ = ["TextReRank", "AioTextReRank"]


def _build_rerank_request(
    model: str,
    query: str,
    documents: List[str],
    return_documents: bool = None,
    top_n: int = None,
    instruct: str = None,
    **kwargs,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    if query is None or documents is None or not documents:
        raise InputRequired("query and documents are required!")
    if model is None or not model:
        raise ModelRequired("Model is required!")

    task_group, function = _get_task_group_and_task(__name__)
    rerank_input = {
        "query": query,
        "documents": documents,
    }
    parameters = {}
    if return_documents is not None:
        parameters["return_documents"] = return_documents
    if top_n is not None:
        parameters["top_n"] = top_n
    if instruct is not None:
        parameters["instruct"] = instruct
    parameters = {**parameters, **kwargs}

    return task_group, function, rerank_input, parameters


class TextReRank(BaseApi):
    task = "text-rerank"
    """API for rerank models.

    """

    class Models:
        gte_rerank = "gte-rerank"
        gte_rerank_v2 = "gte-rerank-v2"
        qwen3_rerank = "qwen3-rerank"
        qwen3_vl_rerank = "qwen3-vl-rerank"

    @classmethod
    def call(  # type: ignore[override]  # pylint: disable=arguments-renamed
        cls,
        model: str,
        query: str,
        documents: List[str],
        return_documents: bool = None,
        top_n: int = None,
        api_key: str = None,
        instruct: str = None,
        **kwargs,
    ) -> ReRankResponse:
        """Calling rerank service.

        Args:
            model (str): The model to use.
            query (str): The query string.
            documents (List[str]): The documents to rank.
            return_documents(bool, `optional`): enable return origin documents,
                system default is false.
            top_n(int, `optional`): how many documents to return,
                default return all the documents.
            api_key (str, optional): The DashScope api key. Defaults to None.
            instruct (str, optional): Custom task instruction to guide
                ranking strategy. English recommended.

        Raises:
            InputRequired: The query and documents are required.
            ModelRequired: The model is required.

        Returns:
            RerankResponse: The rerank result.
        """

        task_group, function, rerank_input, parameters = _build_rerank_request(
            model=model,
            query=query,
            documents=documents,
            return_documents=return_documents,
            top_n=top_n,
            instruct=instruct,
            **kwargs,
        )

        response = super().call(
            model=model,
            task_group=task_group,
            task=TextReRank.task,
            function=function,
            api_key=api_key,
            input=rerank_input,
            **parameters,  # type: ignore[arg-type]
        )

        return ReRankResponse.from_api_response(response)


class AioTextReRank(BaseAioApi):
    task = "text-rerank"
    """Async API for rerank models."""

    Models = TextReRank.Models

    @classmethod
    # pylint: disable=arguments-renamed
    async def call(  # type: ignore[override]
        cls,
        model: str,
        query: str,
        documents: List[str],
        return_documents: bool = None,
        top_n: int = None,
        api_key: str = None,
        workspace: str = None,
        instruct: str = None,
        **kwargs,
    ) -> ReRankResponse:
        """Calling rerank service asynchronously.

        Args:
            model (str): The model to use.
            query (str): The query string.
            documents (List[str]): The documents to rank.
            return_documents(bool, `optional`): enable return origin documents,
                system default is false.
            top_n(int, `optional`): how many documents to return,
                default return all the documents.
            api_key (str, optional): The DashScope api key. Defaults to None.
            workspace (str, optional): The DashScope workspace id.
            instruct (str, optional): Custom task instruction to guide
                ranking strategy. English recommended.

        Raises:
            InputRequired: The query and documents are required.
            ModelRequired: The model is required.

        Returns:
            RerankResponse: The rerank result.
        """
        task_group, function, rerank_input, parameters = _build_rerank_request(
            model=model,
            query=query,
            documents=documents,
            return_documents=return_documents,
            top_n=top_n,
            instruct=instruct,
            **kwargs,
        )

        response = await super().call(
            model=model,
            task_group=task_group,
            task=AioTextReRank.task,
            function=function,
            api_key=api_key,
            workspace=workspace,
            input=rerank_input,
            **parameters,  # type: ignore[arg-type]
        )

        return ReRankResponse.from_api_response(response)
