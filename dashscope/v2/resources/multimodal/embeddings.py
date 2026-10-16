# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.multimodal.embeddings resource (design doc M3 3.2)."""

# "input" is the public API parameter name (mirrors OpenAI embeddings).
# pylint: disable=redefined-builtin

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from dashscope.v2.errors import check_response
from dashscope.v2.resources._oss import upload_embedding_media
from dashscope.v2.resources.embeddings import (
    EmbeddingResponse,
    parse_embedding_response,
)
from dashscope.v2.types.common import with_headers

__all__ = [
    "AsyncMultimodalEmbeddings",
    "EmbeddingItem",
    "MultimodalEmbeddings",
]

_PATH = "/services/embeddings/multimodal-embedding/multimodal-embedding"

MODULE = "embeddings"

# The service only fetches oss:// URLs when the request opts in.
_OSS_RESOLVE_HEADERS = {"X-DashScope-OssResourceResolve": "enable"}


class EmbeddingItem(BaseModel):
    """One multimodal input element; exactly one modality field is set."""

    model_config = ConfigDict(extra="allow")

    text: Optional[str] = None
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None
    factor: Optional[float] = None


_MODALITY_KEYS = ("text", "image", "audio", "video")


def _contents(
    input: Union[Dict[str, Any], EmbeddingItem, List[Any]],
) -> List[Dict[str, Any]]:
    single = isinstance(input, (dict, EmbeddingItem))
    items = [input] if single else list(input)
    if not items:
        raise ValueError("input must not be empty")
    contents: List[Dict[str, Any]] = []
    for item in items:
        data = (
            item.model_dump(exclude_none=True)
            if isinstance(item, EmbeddingItem)
            else dict(item)
        )
        if not any(data.get(key) is not None for key in _MODALITY_KEYS):
            raise ValueError(
                "each input item needs one of text/image/audio/video",
            )
        contents.append(data)
    return contents


def _uses_oss(contents: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(value, str) and value.startswith("oss://")
        for item in contents
        for value in item.values()
    )


def _build_body(
    *,
    model: str,
    contents: List[Dict[str, Any]],
    dimension: Optional[int],
    output_type: Optional[str],
    fps: Optional[float],
    instruct: Optional[str],
    enable_fusion: Optional[bool],
    res_level: Optional[int],
    max_video_frames: Optional[int],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key, value in (
        ("dimension", dimension),
        ("output_type", output_type),
        ("fps", fps),
        ("instruct", instruct),
        ("enable_fusion", enable_fusion),
        ("res_level", res_level),
        ("max_video_frames", max_video_frames),
    ):
        if value is not None:
            merged[key] = value
    if parameters:
        merged.update(parameters)
    return {
        "model": model,
        "input": {"contents": contents},
        "parameters": merged,
    }


class MultimodalEmbeddings:
    """Embed text, image, audio and video into one shared vector space."""

    def __init__(self, client) -> None:
        self._client = client

    def create(  # pylint: disable=too-many-arguments
        self,
        *,
        model: str,
        input: Union[Dict[str, Any], EmbeddingItem, List[Any]],
        dimension: Optional[int] = None,
        output_type: Optional[str] = None,
        fps: Optional[float] = None,
        instruct: Optional[str] = None,
        enable_fusion: Optional[bool] = None,
        res_level: Optional[int] = None,
        max_video_frames: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Embed one item, or a list of them, into one vector space.

        Each item sets exactly one of ``text`` / ``image`` / ``audio`` /
        ``video`` (plus an optional ``factor``). ``fps``, ``res_level``
        (resolution tier 0-3) and ``max_video_frames`` apply to snapshot
        models only; anything this signature does not name goes in
        ``parameters``. Local paths and ``file://`` URIs upload to OSS
        first; ``oss://`` inputs are resolved by the service without any
        extra opt-in.
        """
        contents = _contents(input)
        upload_embedding_media(
            self._client,
            contents,
            model,
            api_key,
            workspace,
        )
        body = _build_body(
            model=model,
            contents=contents,
            dimension=dimension,
            output_type=output_type,
            fps=fps,
            instruct=instruct,
            enable_fusion=enable_fusion,
            res_level=res_level,
            max_video_frames=max_video_frames,
            parameters=parameters,
        )
        response = self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            headers=_OSS_RESOLVE_HEADERS if _uses_oss(contents) else None,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            parse_embedding_response(response.json()),
            response,
        )


class AsyncMultimodalEmbeddings:
    def __init__(self, client) -> None:
        self._client = client

    async def create(  # pylint: disable=too-many-arguments
        self,
        *,
        model: str,
        input: Union[Dict[str, Any], EmbeddingItem, List[Any]],
        dimension: Optional[int] = None,
        output_type: Optional[str] = None,
        fps: Optional[float] = None,
        instruct: Optional[str] = None,
        enable_fusion: Optional[bool] = None,
        res_level: Optional[int] = None,
        max_video_frames: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> EmbeddingResponse:
        contents = _contents(input)
        # The upload path is blocking (requests + getPolicy); run it off the
        # event loop so it never stalls other tasks.
        await asyncio.to_thread(
            upload_embedding_media,
            self._client,
            contents,
            model,
            api_key,
            workspace,
        )
        body = _build_body(
            model=model,
            contents=contents,
            dimension=dimension,
            output_type=output_type,
            fps=fps,
            instruct=instruct,
            enable_fusion=enable_fusion,
            res_level=res_level,
            max_video_frames=max_video_frames,
            parameters=parameters,
        )
        response = await self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            headers=_OSS_RESOLVE_HEADERS if _uses_oss(contents) else None,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            parse_embedding_response(response.json()),
            response,
        )
