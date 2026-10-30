# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from typing import List

from dashscope.legacy_1x.api_entities.dashscope_response import (
    DashScopeAPIResponse,
    DictMixin,
)
from dashscope.legacy_1x.client.base_api import BaseApi, BaseAioApi
from dashscope.legacy_1x.common.error import InputRequired, ModelRequired
from dashscope.legacy_1x.common.utils import _get_task_group_and_task
from dashscope.legacy_1x.utils.oss_utils import preprocess_message_element


@dataclass(init=False)
class MultiModalEmbeddingItemBase(DictMixin):
    factor: float

    def __init__(self, factor: float, **kwargs):
        super().__init__(factor=factor, **kwargs)


@dataclass(init=False)
class MultiModalEmbeddingItemText(MultiModalEmbeddingItemBase):
    text: str

    def __init__(self, text: str, factor: float, **kwargs):
        super().__init__(factor, **kwargs)
        self.text = text


@dataclass(init=False)
class MultiModalEmbeddingItemImage(MultiModalEmbeddingItemBase):
    image: str

    def __init__(self, image: str, factor: float, **kwargs):
        super().__init__(factor, **kwargs)
        self.image = image


@dataclass(init=False)
class MultiModalEmbeddingItemAudio(MultiModalEmbeddingItemBase):
    audio: str

    def __init__(self, audio: str, factor: float, **kwargs):
        super().__init__(factor, **kwargs)
        self.audio = audio


class MultiModalEmbedding(BaseApi):
    task = "multimodal-embedding"

    class Models:
        multimodal_embedding_one_peace_v1 = "multimodal-embedding-one-peace-v1"
        multimodal_embedding_v1 = "multimodal-embedding-v1"
        qwen3_vl_embedding = "qwen3-vl-embedding"
        qwen2_5_vl_embedding = "qwen2.5-vl-embedding"
        tongyi_embedding_vision_plus = "tongyi-embedding-vision-plus"
        tongyi_embedding_vision_flash = "tongyi-embedding-vision-flash"

    @classmethod
    def call(  # type: ignore[override]  # pylint: disable=arguments-renamed
        cls,
        model: str,
        # pylint: disable=redefined-builtin
        input: List[MultiModalEmbeddingItemBase],
        api_key: str = None,
        workspace: str = None,
        dimension: int = None,
        output_type: str = None,
        fps: float = None,
        instruct: str = None,
        enable_fusion: bool = None,
        res_level: int = None,
        max_video_frames: int = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Get embedding multimodal contents..

        Args:
            model (str): The embedding model name.
            input (List[MultiModalEmbeddingElement]): The embedding
                elements, every element include data, modal, factor field.
            workspace (str): The dashscope workspace id.
            dimension (int, optional): Output vector dimensions.
                Model-specific supported values.
            output_type (str, optional): Output vector format,
                currently only "dense" is supported.
            fps (float, optional): Video frame extraction ratio
                in range [0,1]. Default: 1.0.
            instruct (str, optional): Custom task instruction to guide
                model understanding of query intent.
            enable_fusion (bool, optional): Only for qwen3-vl-embedding.
                When True, fuses all contents into 1 vector.
            res_level (int, optional): Resolution tier: 0/1/2/3.
                Only for snapshot models.
            max_video_frames (int, optional): Max video sampling frames,
                up to 64. Only for snapshot models.
            **kwargs:
                auto_truncation(bool, `optional`): Automatically truncate
                audio longer than 15 seconds or text longer than 70 words.
                Default to false(Too long input will result in failure).

        Returns:
            DashScopeAPIResponse: The embedding result.
        """
        if input is None or not input:
            raise InputRequired("prompt is required!")
        if model is None or not model:
            raise ModelRequired("Model is required!")
        embedding_input = {}
        has_upload = cls._preprocess_message_inputs(
            model,
            input,  # type: ignore[arg-type]
            api_key,
        )  # noqa: E501
        if has_upload:
            headers = kwargs.pop("headers", {})
            headers["X-DashScope-OssResourceResolve"] = "enable"
            kwargs["headers"] = headers
        embedding_input["contents"] = input
        kwargs.pop("stream", False)  # not support streaming output.
        if dimension is not None:
            kwargs["dimension"] = dimension
        if output_type is not None:
            kwargs["output_type"] = output_type
        if fps is not None:
            kwargs["fps"] = fps
        if instruct is not None:
            kwargs["instruct"] = instruct
        if enable_fusion is not None:
            kwargs["enable_fusion"] = enable_fusion
        if res_level is not None:
            kwargs["res_level"] = res_level
        if max_video_frames is not None:
            kwargs["max_video_frames"] = max_video_frames
        task_group, function = _get_task_group_and_task(__name__)
        return super().call(
            model=model,
            input=embedding_input,
            task_group=task_group,
            task=MultiModalEmbedding.task,
            function=function,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )

    @classmethod
    def _preprocess_message_inputs(
        cls,
        model: str,
        input_data: List[dict],
        api_key: str,
    ):
        """preprocess following inputs
        input = [{'factor': 1, 'text': 'hello'},
                {'factor': 2, 'audio': ''},
                {'factor': 3, 'image': ''}]
        """
        has_upload = False
        upload_certificate = None
        for elem in input_data:
            if not isinstance(elem, (int, float, bool, str, bytes, bytearray)):
                is_upload, upload_certificate = preprocess_message_element(
                    model,
                    elem,
                    api_key,
                    upload_certificate,  # type: ignore[arg-type]
                )
                if is_upload and not has_upload:
                    has_upload = True
        return has_upload


class AioMultiModalEmbedding(BaseAioApi):
    task = "multimodal-embedding"

    class Models:
        multimodal_embedding_one_peace_v1 = "multimodal-embedding-one-peace-v1"
        multimodal_embedding_v1 = "multimodal-embedding-v1"
        qwen3_vl_embedding = "qwen3-vl-embedding"
        qwen2_5_vl_embedding = "qwen2.5-vl-embedding"
        tongyi_embedding_vision_plus = "tongyi-embedding-vision-plus"
        tongyi_embedding_vision_flash = "tongyi-embedding-vision-flash"

    @classmethod
    async def call(  # type: ignore[override]  # pylint: disable=arguments-renamed  # noqa: E501
        cls,
        model: str,
        # pylint: disable=redefined-builtin
        input: List[MultiModalEmbeddingItemBase],
        api_key: str = None,
        workspace: str = None,
        dimension: int = None,
        output_type: str = None,
        fps: float = None,
        instruct: str = None,
        enable_fusion: bool = None,
        res_level: int = None,
        max_video_frames: int = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Get embedding multimodal contents..

        Args:
            model (str): The embedding model name.
            input (List[MultiModalEmbeddingElement]): The embedding
                elements, every element include data, modal, factor field.
            workspace (str): The dashscope workspace id.
            dimension (int, optional): Output vector dimensions.
                Model-specific supported values.
            output_type (str, optional): Output vector format,
                currently only "dense" is supported.
            fps (float, optional): Video frame extraction ratio
                in range [0,1]. Default: 1.0.
            instruct (str, optional): Custom task instruction to guide
                model understanding of query intent.
            enable_fusion (bool, optional): Only for qwen3-vl-embedding.
                When True, fuses all contents into 1 vector.
            res_level (int, optional): Resolution tier: 0/1/2/3.
                Only for snapshot models.
            max_video_frames (int, optional): Max video sampling frames,
                up to 64. Only for snapshot models.
            **kwargs:
                auto_truncation(bool, `optional`): Automatically truncate
                audio longer than 15 seconds or text longer than 70 words.
                Default to false(Too long input will result in failure).

        Returns:
            DashScopeAPIResponse: The embedding result.
        """
        if input is None or not input:
            raise InputRequired("prompt is required!")
        if model is None or not model:
            raise ModelRequired("Model is required!")
        embedding_input = {}
        has_upload = cls._preprocess_message_inputs(
            model,
            input,  # type: ignore[arg-type]
            api_key,
        )  # noqa: E501
        if has_upload:
            headers = kwargs.pop("headers", {})
            headers["X-DashScope-OssResourceResolve"] = "enable"
            kwargs["headers"] = headers
        embedding_input["contents"] = input
        kwargs.pop("stream", False)  # not support streaming output.
        if dimension is not None:
            kwargs["dimension"] = dimension
        if output_type is not None:
            kwargs["output_type"] = output_type
        if fps is not None:
            kwargs["fps"] = fps
        if instruct is not None:
            kwargs["instruct"] = instruct
        if enable_fusion is not None:
            kwargs["enable_fusion"] = enable_fusion
        if res_level is not None:
            kwargs["res_level"] = res_level
        if max_video_frames is not None:
            kwargs["max_video_frames"] = max_video_frames
        task_group, function = _get_task_group_and_task(__name__)
        response = await super().call(
            model=model,
            input=embedding_input,
            task_group=task_group,
            task=MultiModalEmbedding.task,
            function=function,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )
        return response

    @classmethod
    def _preprocess_message_inputs(
        cls,
        model: str,
        input_data: List[dict],
        api_key: str,
    ):
        """preprocess following inputs
        input = [{'factor': 1, 'text': 'hello'},
                {'factor': 2, 'audio': ''},
                {'factor': 3, 'image': ''}]
        """
        has_upload = False
        upload_certificate = None
        for elem in input_data:
            if not isinstance(elem, (int, float, bool, str, bytes, bytearray)):
                is_upload, upload_certificate = preprocess_message_element(
                    model,
                    elem,
                    api_key,
                    upload_certificate,  # type: ignore[arg-type]
                )
                if is_upload and not has_upload:
                    has_upload = True
        return has_upload
