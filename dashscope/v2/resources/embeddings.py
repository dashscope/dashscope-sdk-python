# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.embeddings resource (native text embedding + batch jobs).

Multimodal embedding lives at ``client.multimodal.embeddings``.
"""

# "input" is the public API parameter name (mirrors OpenAI embeddings).
# pylint: disable=redefined-builtin

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dashscope.v2.errors import check_response
from dashscope.v2.resources._jobs import (
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    AsyncJobs,
    Job,
    JobList,
    Jobs,
)
from dashscope.v2.types.common import (
    ResponseBase,
    Usage,
    response_envelope,
    with_headers,
)

_PATH = "/services/embeddings/text-embedding/text-embedding"
_BATCH_SERVICE = "embeddings/text-embedding/text-embedding"
# Functional-area tag for x-dashscope-sdk-client (matches 1.x ``embeddings``).
MODULE = "embeddings"

__all__ = [
    "AsyncEmbeddings",
    "Embedding",
    "EmbeddingResponse",
    "Embeddings",
]


class Embedding(BaseModel):
    """One embedding vector; ``type`` says which modality produced it."""

    model_config = ConfigDict(extra="allow")

    # The text endpoint reports the position as "text_index"; every other
    # endpoint calls it "index". One public name, both wire spellings.
    index: int = Field(0, validation_alias=AliasChoices("index", "text_index"))
    type: Optional[str] = None
    embedding: List[float] = []


class EmbeddingResponse(ResponseBase):
    embeddings: List[Embedding] = []
    usage: Optional[Usage] = None


def parse_embedding_response(body: Dict[str, Any]) -> EmbeddingResponse:
    """Shared by the text and multimodal endpoints: one response shape."""
    output = body.get("output") or {}
    embeddings = output.get("embeddings") or []
    if not embeddings and output.get("embedding"):
        # multimodal-embedding-one-peace-v1 predates the plural list: it
        # fuses every content into one vector returned as output.embedding.
        # Surface it as item 0 of the canonical list so callers read one
        # shape regardless of model generation. 1.x exposed the raw key via
        # response.output["embedding"]; v2 normalizes instead.
        embeddings = [{"index": 0, "embedding": output["embedding"]}]
    return EmbeddingResponse(
        embeddings=embeddings,
        usage=body.get("usage"),
        **response_envelope(body),
    )


def _build_body(
    *,
    model: str,
    input: Union[str, List[str]],
    text_type: Optional[str],
    dimension: Optional[int],
    output_type: Optional[str],
    instruct: Optional[str],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    texts = [input] if isinstance(input, str) else list(input)
    if not texts:
        raise ValueError("input must not be empty")
    merged: Dict[str, Any] = {}
    for key, value in (
        ("text_type", text_type),
        ("dimension", dimension),
        ("output_type", output_type),
        ("instruct", instruct),
    ):
        if value is not None:
            merged[key] = value
    if parameters:
        merged.update(parameters)
    return {
        "model": model,
        "input": {"texts": texts},
        "parameters": merged,
    }


class EmbeddingJobs:
    """Async batch embedding task facade over the Jobs infra."""

    def __init__(self, client) -> None:
        self._jobs = Jobs(client, _BATCH_SERVICE, MODULE)

    def create(
        self,
        *,
        model: str,
        file_url: str,
        text_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Submit a batch task over one file whose lines are the texts.

        Singular, unlike ``asr.transcribe(file_urls=)``: this endpoint
        accepts exactly one file.
        """
        if not file_url:
            raise ValueError("file_url must not be empty")
        merged = dict(parameters or {})
        if text_type is not None:
            merged["text_type"] = text_type
        return self._jobs.create(
            model=model,
            input={"url": file_url},
            parameters=merged or None,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return self._jobs.fetch(
            task_id,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Cancel a PENDING task (``POST /tasks/{task_id}/cancel``)."""
        return self._jobs.cancel(
            task_id,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key_id: Optional[str] = None,
        region: Optional[str] = None,
        status: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobList:
        """List async tasks (``GET /tasks``)."""
        return self._jobs.list(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
            api_key_id=api_key_id,
            region=region,
            status=status,
            page_no=page_no,
            page_size=page_size,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    def wait(
        self,
        job: Union[Job, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> Job:
        return self._jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            _sleep=_sleep,
        )


class AsyncEmbeddingJobs:
    def __init__(self, client) -> None:
        self._jobs = AsyncJobs(client, _BATCH_SERVICE, MODULE)

    async def create(
        self,
        *,
        model: str,
        file_url: str,
        text_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        if not file_url:
            raise ValueError("file_url must not be empty")
        merged = dict(parameters or {})
        if text_type is not None:
            merged["text_type"] = text_type
        return await self._jobs.create(
            model=model,
            input={"url": file_url},
            parameters=merged or None,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return await self._jobs.fetch(
            task_id,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    async def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Cancel a PENDING task (``POST /tasks/{task_id}/cancel``)."""
        return await self._jobs.cancel(
            task_id,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key_id: Optional[str] = None,
        region: Optional[str] = None,
        status: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobList:
        """List async tasks (``GET /tasks``)."""
        return await self._jobs.list(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
            api_key_id=api_key_id,
            region=region,
            status=status,
            page_no=page_no,
            page_size=page_size,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
        )

    async def wait(
        self,
        job: Union[Job, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        _sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> Job:
        return await self._jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            _sleep=_sleep,
        )


class Embeddings:
    def __init__(self, client) -> None:
        self._client = client
        self.jobs = EmbeddingJobs(client)

    def create(
        self,
        *,
        model: str,
        input: Union[str, List[str]],
        text_type: Optional[str] = None,
        dimension: Optional[int] = None,
        output_type: Optional[str] = None,
        instruct: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> EmbeddingResponse:
        body = _build_body(
            model=model,
            input=input,
            text_type=text_type,
            dimension=dimension,
            output_type=output_type,
            instruct=instruct,
            parameters=parameters,
        )
        response = self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            parse_embedding_response(response.json()),
            response,
        )


class AsyncEmbeddings:
    def __init__(self, client) -> None:
        self._client = client
        self.jobs = AsyncEmbeddingJobs(client)

    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[str]],
        text_type: Optional[str] = None,
        dimension: Optional[int] = None,
        output_type: Optional[str] = None,
        instruct: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> EmbeddingResponse:
        body = _build_body(
            model=model,
            input=input,
            text_type=text_type,
            dimension=dimension,
            output_type=output_type,
            instruct=instruct,
            parameters=parameters,
        )
        response = await self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            parse_embedding_response(response.json()),
            response,
        )
