# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Union

from dashscope.legacy_1x.api_entities.dashscope_response import (
    DashScopeAPIResponse,
    ImageSynthesisResponse,
)
from dashscope.legacy_1x.client.base_api import (
    BaseAsyncApi,
    BaseApi,
    BaseAsyncAioApi,
    BaseAioApi,
)
from dashscope.legacy_1x.common.constants import (
    IMAGES,
    NEGATIVE_PROMPT,
    PROMPT,
)
from dashscope.legacy_1x.common.error import InputRequired
from dashscope.legacy_1x.common.utils import _get_task_group_and_task
from dashscope.legacy_1x.utils.oss_utils import check_and_upload_local


class ImageSynthesis(BaseAsyncApi):
    task = "text2image"
    """API for image synthesis.
    """

    class Models:
        wanx_v1 = "wanx-v1"
        wanx_sketch_to_image_v1 = "wanx-sketch-to-image-v1"

        wanx_2_1_imageedit = "wanx2.1-imageedit"

    @classmethod
    def call(  # type: ignore[override]
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        workspace: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        size: str = None,
        n: int = None,
        seed: int = None,
        style: str = None,
        ref_strength: float = None,
        ref_mode: str = None,
        prompt_extend: bool = None,
        watermark: bool = None,
        bbox_list: List = None,
        enable_sequential: bool = None,
        thinking_mode: str = None,
        color_palette: str = None,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Call image(s) synthesis service and get result.

        Args:
            model (str): The model, reference ``Models``.
            prompt (Any): The prompt for image(s) synthesis.
            negative_prompt (Any): The negative_prompt. Defaults to None.
            images (List[str]): The input list of images url,
                currently not supported.
            api_key (str, optional): The api api_key. Defaults to None.
            sketch_image_url (str, optional): Only for wanx-sketch-to-image-v1,
                can be local file.
                Defaults to None.
            workspace (str): The dashscope workspace id.
            extra_input (Dict): The extra input parameters.
            task (str): The task of api, ref doc.
            function (str): The specific functions to be achieved. like:
                colorization,super_resolution,expand,remove_watermaker,doodle,
                description_edit_with_mask,description_edit,stylization_local,stylization_all
            base_image_url (str): Enter the URL address of the target edited image.
            mask_image_url (str): Provide the URL address of the image of the marked area by the user. It should be consistent with the image resolution of the base_image_url.  # pylint: disable=line-too-long
            size (str, optional): The output image(s) size(width*height).
            n (int, optional): Number of images to synthesis.
            seed (int, optional): Random seed for image generation.
            style (str, optional): The output image style. Supported values:
                <auto>, <photography>, <portrait>, <3d cartoon>, <anime>,
                <oil painting>, <watercolor>, <sketch>, <chinese painting>,
                <flat illustration>. Default is <auto>.
            ref_strength (float, optional): Control the similarity between
                output image and reference image. Range [0.0, 1.0].
                Higher value means more similar to reference image.
            ref_mode (str, optional): Mode for generating image based on
                reference image. Supported values: repaint (default, based on
                content), refonly (based on style).
            prompt_extend (bool, optional): Whether to extend prompt
                automatically for better results.
            watermark (bool, optional): Whether to add watermark.
            bbox_list (List, optional): List of bounding boxes for region-specific editing.
            enable_sequential (bool, optional): Enable sequential generation mode.
            thinking_mode (str, optional): Thinking mode for generation. Supported values: fast, balanced, quality.
            color_palette (str, optional): Color palette specification for consistent styling.
            **kwargs:
                similarity(float, `optional`): The similarity between the
                    output image and the input image
                sketch_weight(int, optional): How much the input sketch
                    affects the output image[0-10], only for wanx-sketch-to-image-v1. # noqa E501
                    Default 10.
                realisticness(int, optional): The realisticness of the output
                    image[0-10], only for wanx-sketch-to-image-v1. Default 5

        Raises:
            InputRequired: The prompt cannot be empty.

        Returns:
            ImageSynthesisResponse: The image(s) synthesis result.
        """
        if size is not None:
            kwargs["size"] = size
        if n is not None:
            kwargs["n"] = n
        if seed is not None:
            kwargs["seed"] = seed
        if style is not None:
            kwargs["style"] = style
        if ref_strength is not None:
            kwargs["ref_strength"] = ref_strength
        if ref_mode is not None:
            kwargs["ref_mode"] = ref_mode
        if prompt_extend is not None:
            kwargs["prompt_extend"] = prompt_extend
        if watermark is not None:
            kwargs["watermark"] = watermark
        if bbox_list is not None:
            kwargs["bbox_list"] = bbox_list
        if enable_sequential is not None:
            kwargs["enable_sequential"] = enable_sequential
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        if color_palette is not None:
            kwargs["color_palette"] = color_palette
        return super().call(  # type: ignore[return-value]
            model,
            prompt,
            negative_prompt,
            images,
            api_key=api_key,
            sketch_image_url=sketch_image_url,
            ref_img=ref_img,
            workspace=workspace,
            extra_input=extra_input,
            task=task,
            function=function,
            mask_image_url=mask_image_url,
            base_image_url=base_image_url,
            **kwargs,
        )

    @classmethod
    def sync_call(
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        workspace: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        size: str = None,
        n: int = None,
        seed: int = None,
        style: str = None,
        ref_strength: float = None,
        ref_mode: str = None,
        prompt_extend: bool = None,
        watermark: bool = None,
        bbox_list: List = None,
        enable_sequential: bool = None,
        thinking_mode: str = None,
        color_palette: str = None,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """
        Note: This method currently now only supports wan2.2-t2i-flash and wan2.2-t2i-plus.  # noqa: E501  # pylint: disable=line-too-long
            Using other models will result in an error，More raw image models may be added for use later  # pylint: disable=line-too-long
        """
        if size is not None:
            kwargs["size"] = size
        if n is not None:
            kwargs["n"] = n
        if seed is not None:
            kwargs["seed"] = seed
        if style is not None:
            kwargs["style"] = style
        if ref_strength is not None:
            kwargs["ref_strength"] = ref_strength
        if ref_mode is not None:
            kwargs["ref_mode"] = ref_mode
        if prompt_extend is not None:
            kwargs["prompt_extend"] = prompt_extend
        if watermark is not None:
            kwargs["watermark"] = watermark
        if bbox_list is not None:
            kwargs["bbox_list"] = bbox_list
        if enable_sequential is not None:
            kwargs["enable_sequential"] = enable_sequential
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        if color_palette is not None:
            kwargs["color_palette"] = color_palette
        task_group, f = _get_task_group_and_task(__name__)
        inputs, kwargs, task = cls._get_input(
            model,
            prompt,
            negative_prompt,
            images,
            api_key,
            sketch_image_url,
            ref_img,
            extra_input,
            task,
            function,
            mask_image_url,
            base_image_url,
            **kwargs,
        )
        response = BaseApi.call(
            model,
            inputs,
            task_group,
            task,
            f,
            api_key,
            workspace,
            **kwargs,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    def _get_input(  # pylint: disable=too-many-branches
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        **kwargs,
    ):
        if prompt is None or not prompt:
            raise InputRequired("prompt is required!")
        inputs = {PROMPT: prompt}
        has_upload = False
        upload_certificate = None

        if negative_prompt is not None:
            inputs[NEGATIVE_PROMPT] = negative_prompt
        if images is not None and images and len(images) > 0:
            new_images = []
            for image in images:
                (
                    is_upload,
                    new_image,
                    upload_certificate,
                ) = check_and_upload_local(
                    model,
                    image,
                    api_key,
                    upload_certificate,  # type: ignore[arg-type]
                )
                if is_upload:
                    has_upload = True
                new_images.append(new_image)
            inputs[IMAGES] = new_images
        if sketch_image_url is not None and sketch_image_url:
            (
                is_upload,
                sketch_image_url,
                upload_certificate,
            ) = check_and_upload_local(
                model,
                sketch_image_url,
                api_key,
                upload_certificate,  # type: ignore[arg-type]
            )
            if is_upload:
                has_upload = True
            inputs["sketch_image_url"] = sketch_image_url
        if ref_img is not None and ref_img:
            is_upload, ref_img, upload_certificate = check_and_upload_local(
                model,
                ref_img,
                api_key,
                upload_certificate,  # type: ignore[arg-type]
            )
            if is_upload:
                has_upload = True
            inputs["ref_img"] = ref_img

        if function is not None and function:
            inputs["function"] = function

        if mask_image_url is not None and mask_image_url:
            (
                is_upload,
                res_mask_image_url,
                upload_certificate,
            ) = check_and_upload_local(
                model,
                mask_image_url,
                api_key,
                upload_certificate,  # type: ignore[arg-type]
            )
            if is_upload:
                has_upload = True
            inputs["mask_image_url"] = res_mask_image_url

        if base_image_url is not None and base_image_url:
            (
                is_upload,
                res_base_image_url,
                upload_certificate,
            ) = check_and_upload_local(
                model,
                base_image_url,
                api_key,
                upload_certificate,  # type: ignore[arg-type]
            )
            if is_upload:
                has_upload = True
            inputs["base_image_url"] = res_base_image_url

        if extra_input is not None and extra_input:
            inputs = {**inputs, **extra_input}

        if has_upload:
            headers = kwargs.pop("headers", {})
            headers["X-DashScope-OssResourceResolve"] = "enable"
            kwargs["headers"] = headers

        def __get_i2i_task(task, model) -> str:
            # Handle task parameter: prefer valid task value
            if task is not None and task != "":
                return task

            # Determine task type based on model
            if model is not None and model != "":
                if "imageedit" in model or "wan2.5-i2i" in model:
                    return "image2image"

            # Default to text-to-image task
            return ImageSynthesis.task

        task = __get_i2i_task(task, model)

        return inputs, kwargs, task

    @classmethod
    # type: ignore[override]
    def async_call(  # pylint: disable=arguments-renamed  # type: ignore[override] # noqa: E501
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        workspace: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        size: str = None,
        n: int = None,
        seed: int = None,
        style: str = None,
        ref_strength: float = None,
        ref_mode: str = None,
        prompt_extend: bool = None,
        watermark: bool = None,
        bbox_list: List = None,
        enable_sequential: bool = None,
        thinking_mode: str = None,
        color_palette: str = None,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Create a image(s) synthesis task, and return task information.

        Args:
            model (str): The model, reference ``Models``.
            prompt (Any): The prompt for image(s) synthesis.
            negative_prompt (Any): The negative_prompt. Defaults to None.
            images (List[str]): The input list of images url.
            api_key (str, optional): The api api_key. Defaults to None.
            sketch_image_url (str, optional): Only for wanx-sketch-to-image-v1.
                Defaults to None.
            workspace (str): The dashscope workspace id.
            extra_input (Dict): The extra input parameters.
            task (str): The task of api, ref doc.
            function (str): The specific functions to be achieved.
            mask_image_url (str): Provide the URL address of the image of
                the marked area by the user.
            base_image_url (str): Enter the URL address of the target edited
                image.
            size (str, optional): The output image(s) size(width*height).
            seed (int, optional): Random seed for image generation.
            style (str, optional): The output image style. Supported values:
                <auto>, <photography>, <portrait>, <3d cartoon>, <anime>,
                <oil painting>, <watercolor>, <sketch>, <chinese painting>,
                <flat illustration>. Default is <auto>.
            ref_strength (float, optional): Control the similarity between
                output image and reference image. Range [0.0, 1.0].
            ref_mode (str, optional): Mode for generating image based on
                reference image. Supported values: repaint (default, based on
                content), refonly (based on style).
            prompt_extend (bool, optional): Whether to extend prompt
                automatically for better results.
            watermark (bool, optional): Whether to add watermark.
            bbox_list (List, optional): List of bounding boxes for
                region-specific editing.
            enable_sequential (bool, optional): Enable sequential generation
                mode.
            thinking_mode (str, optional): Thinking mode for generation.
                Supported values: fast, balanced, quality.
            color_palette (str, optional): Color palette specification for
                consistent styling.
        """
        if size is not None:
            kwargs["size"] = size
        if n is not None:
            kwargs["n"] = n
        if seed is not None:
            kwargs["seed"] = seed
        if style is not None:
            kwargs["style"] = style
        if ref_strength is not None:
            kwargs["ref_strength"] = ref_strength
        if ref_mode is not None:
            kwargs["ref_mode"] = ref_mode
        if prompt_extend is not None:
            kwargs["prompt_extend"] = prompt_extend
        if watermark is not None:
            kwargs["watermark"] = watermark
        if bbox_list is not None:
            kwargs["bbox_list"] = bbox_list
        if enable_sequential is not None:
            kwargs["enable_sequential"] = enable_sequential
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        if color_palette is not None:
            kwargs["color_palette"] = color_palette
        task_group, f = _get_task_group_and_task(__name__)
        # pylint: disable=protected-access
        inputs, kwargs, task = ImageSynthesis._get_input(
            model,
            prompt,
            negative_prompt,
            images,
            api_key,
            sketch_image_url,
            ref_img,
            extra_input,
            task,
            function,
            mask_image_url,
            base_image_url,
            **kwargs,
        )
        response = super().async_call(
            model=model,
            task_group=task_group,
            task=task,
            function=f,
            api_key=api_key,
            input=inputs,
            workspace=workspace,
            **kwargs,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    def fetch(  # type: ignore[override]
        cls,
        task: Union[str, ImageSynthesisResponse],
        api_key: str = None,
        workspace: str = None,
    ) -> ImageSynthesisResponse:
        """Fetch image(s) synthesis task status or result.

        Args:
            task (Union[str, ImageSynthesisResponse]): The task_id or
                ImageSynthesisResponse return by async_call().
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            ImageSynthesisResponse: The task status or result.
        """
        response = super().fetch(task, api_key=api_key, workspace=workspace)
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    def wait(  # type: ignore[override]
        cls,
        task: Union[str, ImageSynthesisResponse],
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Wait for image(s) synthesis task to complete, and return the result.

        Args:
            task (Union[str, ImageSynthesisResponse]): The task_id or
                ImageSynthesisResponse return by async_call().
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            ImageSynthesisResponse: The task result.
        """
        response = super().wait(
            task,
            api_key,
            workspace=workspace,
            **kwargs,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    def cancel(  # type: ignore[override]
        cls,
        task: Union[str, ImageSynthesisResponse],
        api_key: str = None,
        workspace: str = None,
    ) -> DashScopeAPIResponse:
        """Cancel image synthesis task.
        Only tasks whose status is PENDING can be canceled.

        Args:
            task (Union[str, ImageSynthesisResponse]): The task_id or
                ImageSynthesisResponse return by async_call().
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            DashScopeAPIResponse: The response data.
        """
        return super().cancel(task, api_key, workspace=workspace)

    @classmethod
    def list(
        cls,
        start_time: str = None,
        end_time: str = None,
        model_name: str = None,
        api_key_id: str = None,
        region: str = None,
        status: str = None,
        page_no: int = 1,
        page_size: int = 10,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """List async tasks.

        Args:
            start_time (str, optional): The tasks start time,
                for example: 20230420000000. Defaults to None.
            end_time (str, optional): The tasks end time,
                for example: 20230420000000. Defaults to None.
            model_name (str, optional): The tasks model name. Defaults to None.
            api_key_id (str, optional): The tasks api-key-id. Defaults to None.
            region (str, optional): The service region,
                for example: cn-beijing. Defaults to None.
            status (str, optional): The status of tasks[PENDING,
                RUNNING, SUCCEEDED, FAILED, CANCELED]. Defaults to None.
            page_no (int, optional): The page number. Defaults to 1.
            page_size (int, optional): The page size. Defaults to 10.
            api_key (str, optional): The user api-key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            DashScopeAPIResponse: The response data.
        """
        return super().list(
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
            **kwargs,
        )


class AioImageSynthesis(BaseAsyncAioApi):
    # type: ignore[override]
    @classmethod
    async def call(  # type: ignore[override] # pylint: disable=arguments-renamed # noqa: E501
        # type: ignore[override]
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        workspace: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        size: str = None,
        n: int = None,
        seed: int = None,
        style: str = None,
        ref_strength: float = None,
        ref_mode: str = None,
        prompt_extend: bool = None,
        watermark: bool = None,
        bbox_list: List = None,
        enable_sequential: bool = None,
        thinking_mode: str = None,
        color_palette: str = None,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Call image(s) synthesis service and get result.

        Args:
            model (str): The model, reference ``Models``.
            prompt (Any): The prompt for image(s) synthesis.
            negative_prompt (Any): The negative_prompt. Defaults to None.
            images (List[str]): The input list of images url,
                currently not supported.
            api_key (str, optional): The api api_key. Defaults to None.
            sketch_image_url (str, optional): Only for wanx-sketch-to-image-v1,
                can be local file.
                Defaults to None.
            workspace (str): The dashscope workspace id.
            extra_input (Dict): The extra input parameters.
            task (str): The task of api, ref doc.
            function (str): The specific functions to be achieved. like:
                colorization,super_resolution,expand,remove_watermaker,doodle,
                description_edit_with_mask,description_edit,stylization_local,stylization_all
            base_image_url (str): Enter the URL address of the target edited image.
            mask_image_url (str): Provide the URL address of the image of the marked area by the user. It should be consistent with the image resolution of the base_image_url.  # pylint: disable=line-too-long
            size (str, optional): The output image(s) size(width*height).
            n (int, optional): Number of images to synthesis.
            seed (int, optional): Random seed for image generation.
            style (str, optional): The output image style. Supported values:
                <auto>, <photography>, <portrait>, <3d cartoon>, <anime>,
                <oil painting>, <watercolor>, <sketch>, <chinese painting>,
                <flat illustration>. Default is <auto>.
            ref_strength (float, optional): Control the similarity between
                output image and reference image. Range [0.0, 1.0].
            ref_mode (str, optional): Mode for generating image based on
                reference image. Supported values: repaint (default, based on
                content), refonly (based on style).
            prompt_extend (bool, optional): Whether to extend prompt
                automatically for better results.
            watermark (bool, optional): Whether to add watermark.
            bbox_list (List, optional): List of bounding boxes for region-specific editing.
            enable_sequential (bool, optional): Enable sequential generation mode.
            thinking_mode (str, optional): Thinking mode for generation. Supported values: fast, balanced, quality.
            color_palette (str, optional): Color palette specification for consistent styling.
            **kwargs:
                similarity(float, `optional`): The similarity between the
                    output image and the input image
                sketch_weight(int, optional): How much the input sketch
                    affects the output image[0-10], only for wanx-sketch-to-image-v1. # noqa E501
                    Default 10.
                realisticness(int, optional): The realisticness of the output
                    image[0-10], only for wanx-sketch-to-image-v1. Default 5

        Raises:
            InputRequired: The prompt cannot be empty.

        Returns:
            ImageSynthesisResponse: The image(s) synthesis result.
        """
        if size is not None:
            kwargs["size"] = size
        if n is not None:
            kwargs["n"] = n
        if seed is not None:
            kwargs["seed"] = seed
        if style is not None:
            kwargs["style"] = style
        if ref_strength is not None:
            kwargs["ref_strength"] = ref_strength
        if ref_mode is not None:
            kwargs["ref_mode"] = ref_mode
        if prompt_extend is not None:
            kwargs["prompt_extend"] = prompt_extend
        if watermark is not None:
            kwargs["watermark"] = watermark
        if bbox_list is not None:
            kwargs["bbox_list"] = bbox_list
        if enable_sequential is not None:
            kwargs["enable_sequential"] = enable_sequential
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        if color_palette is not None:
            kwargs["color_palette"] = color_palette
        task_group, f = _get_task_group_and_task(__name__)
        # pylint: disable=protected-access
        inputs, kwargs, task = ImageSynthesis._get_input(
            model,
            prompt,
            negative_prompt,
            images,
            api_key,
            sketch_image_url,
            ref_img,
            extra_input,
            task,
            function,
            mask_image_url,
            base_image_url,
            **kwargs,
        )
        response = await super().call(
            model,
            inputs,
            task_group,
            task,
            f,
            api_key,
            workspace,
            **kwargs,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    async def sync_call(
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        workspace: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        size: str = None,  # pylint: disable=unused-argument
        n: int = None,  # pylint: disable=unused-argument
        seed: int = None,  # pylint: disable=unused-argument
        style: str = None,  # pylint: disable=unused-argument
        ref_strength: float = None,  # pylint: disable=unused-argument
        ref_mode: str = None,  # pylint: disable=unused-argument
        prompt_extend: bool = None,  # pylint: disable=unused-argument
        watermark: bool = None,  # pylint: disable=unused-argument
        bbox_list: List = None,  # pylint: disable=unused-argument
        enable_sequential: bool = None,  # pylint: disable=unused-argument
        thinking_mode: str = None,  # pylint: disable=unused-argument
        color_palette: str = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> ImageSynthesisResponse:
        """
        Note: This method currently now only supports wan2.2-t2i-flash and wan2.2-t2i-plus.  # noqa: E501  # pylint: disable=line-too-long
            Using other models will result in an error，More raw image models may be added for use later  # pylint: disable=line-too-long
        """
        if size is not None:
            kwargs["size"] = size
        if n is not None:
            kwargs["n"] = n
        if seed is not None:
            kwargs["seed"] = seed
        if style is not None:
            kwargs["style"] = style
        if ref_strength is not None:
            kwargs["ref_strength"] = ref_strength
        if ref_mode is not None:
            kwargs["ref_mode"] = ref_mode
        if prompt_extend is not None:
            kwargs["prompt_extend"] = prompt_extend
        if watermark is not None:
            kwargs["watermark"] = watermark
        if bbox_list is not None:
            kwargs["bbox_list"] = bbox_list
        if enable_sequential is not None:
            kwargs["enable_sequential"] = enable_sequential
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        if color_palette is not None:
            kwargs["color_palette"] = color_palette
        task_group, f = _get_task_group_and_task(__name__)
        # pylint: disable=protected-access
        inputs, kwargs, task = ImageSynthesis._get_input(
            model,
            prompt,
            negative_prompt,
            images,
            api_key,
            sketch_image_url,
            ref_img,
            extra_input,
            task,
            function,
            mask_image_url,
            base_image_url,
            **kwargs,
        )
        response = await BaseAioApi.call(
            model,
            inputs,
            task_group,
            task,
            f,
            api_key,
            workspace,
            **kwargs,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    async def async_call(  # type: ignore[override]  # pylint: disable=arguments-renamed  # noqa: E501
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] = None,
        api_key: str = None,
        sketch_image_url: str = None,
        ref_img: str = None,
        workspace: str = None,
        extra_input: Dict = None,
        task: str = None,
        function: str = None,
        mask_image_url: str = None,
        base_image_url: str = None,
        size: str = None,  # pylint: disable=unused-argument
        n: int = None,  # pylint: disable=unused-argument
        seed: int = None,  # pylint: disable=unused-argument
        style: str = None,  # pylint: disable=unused-argument
        ref_strength: float = None,  # pylint: disable=unused-argument
        ref_mode: str = None,  # pylint: disable=unused-argument
        prompt_extend: bool = None,  # pylint: disable=unused-argument
        watermark: bool = None,  # pylint: disable=unused-argument
        bbox_list: List = None,  # pylint: disable=unused-argument
        enable_sequential: bool = None,  # pylint: disable=unused-argument
        thinking_mode: str = None,  # pylint: disable=unused-argument
        color_palette: str = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Create a image(s) synthesis task, and return task information.

        Note: This method overrides BaseAsyncAioApi.async_call() with
        renamed parameters to provide a more user-friendly API. The
        generic parameters (input_data, task_group) are replaced with
        domain-specific ones (prompt, negative_prompt, images, etc.).
        Pylint's arguments-renamed warning is disabled for this reason.

        Args:
            model (str): The model, reference ``Models``.
            prompt (Any): The prompt for image(s) synthesis.
            negative_prompt (Any): The negative_prompt. Defaults to None.
            images (List[str]): The input list of images url.
            api_key (str, optional): The api api_key. Defaults to None.
            sketch_image_url (str, optional): Only for wanx-sketch-to-image-v1.
                Defaults to None.
            workspace (str): The dashscope workspace id.
            extra_input (Dict): The extra input parameters.
            task (str): The task of api, ref doc.
            function (str): The specific functions to be achieved. like:
                colorization,super_resolution,expand,remove_watermaker,doodle,
                description_edit_with_mask,description_edit,stylization_local,stylization_all
            base_image_url (str): Enter the URL address of the target edited image.
            mask_image_url (str): Provide the URL address of the image of the marked area by the user. It should be consistent with the image resolution of the base_image_url.  # pylint: disable=line-too-long
            **kwargs(wanx-v1):
                n(int, `optional`): Number of images to synthesis.
                size: The output image(s) size, Default 1024*1024
                similarity(float, `optional`): The similarity between the
                    output image and the input image.
                sketch_weight(int, optional): How much the input sketch
                    affects the output image[0-10], only for wanx-sketch-to-image-v1. # noqa E501
                    Default 10.
                realisticness(int, optional): The realisticness of the output
                    image[0-10], only for wanx-sketch-to-image-v1. Default 5

        Raises:
            InputRequired: The prompt cannot be empty.

        Returns:
            DashScopeAPIResponse: The image synthesis
                task id in the response.
        """
        if size is not None:
            kwargs["size"] = size
        if n is not None:
            kwargs["n"] = n
        if seed is not None:
            kwargs["seed"] = seed
        if style is not None:
            kwargs["style"] = style
        if ref_strength is not None:
            kwargs["ref_strength"] = ref_strength
        if ref_mode is not None:
            kwargs["ref_mode"] = ref_mode
        if prompt_extend is not None:
            kwargs["prompt_extend"] = prompt_extend
        if watermark is not None:
            kwargs["watermark"] = watermark
        if bbox_list is not None:
            kwargs["bbox_list"] = bbox_list
        if enable_sequential is not None:
            kwargs["enable_sequential"] = enable_sequential
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        if color_palette is not None:
            kwargs["color_palette"] = color_palette
        task_group, f = _get_task_group_and_task(__name__)
        # pylint: disable=protected-access
        inputs, kwargs, task = ImageSynthesis._get_input(
            model,
            prompt,
            negative_prompt,
            images,
            api_key,
            sketch_image_url,
            ref_img,
            extra_input,
            task,
            function,
            mask_image_url,
            base_image_url,
            **kwargs,
        )
        response = await super().async_call(
            model,
            inputs,
            task_group,
            task,
            f,
            api_key,
            workspace,
            **kwargs,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    async def fetch(
        cls,
        task: Union[str, ImageSynthesisResponse],  # type: ignore[override]
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Fetch image(s) synthesis task status or result.

        Args:
            task (Union[str, ImageSynthesisResponse]): The task_id or
                ImageSynthesisResponse return by async_call().
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            ImageSynthesisResponse: The task status or result.
        """
        response = await super().fetch(
            task,
            api_key=api_key,
            workspace=workspace,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    async def wait(
        cls,
        task: Union[str, ImageSynthesisResponse],  # type: ignore[override]
        api_key: str = None,
        workspace: str = None,
        wait_timeout: int = -1,
        **kwargs,
    ) -> ImageSynthesisResponse:
        """Wait for image(s) synthesis task to complete, and return the result.

        Args:
            task (Union[str, ImageSynthesisResponse]): The task_id or
                ImageSynthesisResponse return by async_call().
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.
            wait_timeout (int, optional): The maximum seconds to wait.
                Default is -1 (no timeout).

        Returns:
            ImageSynthesisResponse: The task result.
        """
        response = await super().wait(
            task,
            api_key,
            workspace=workspace,
            wait_timeout=wait_timeout,
        )
        return ImageSynthesisResponse.from_api_response(response)

    @classmethod
    async def cancel(
        cls,
        task: Union[str, ImageSynthesisResponse],  # type: ignore[override]
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Cancel image synthesis task.
        Only tasks whose status is PENDING can be canceled.

        Args:
            task (Union[str, ImageSynthesisResponse]): The task_id or
                ImageSynthesisResponse return by async_call().
            api_key (str, optional): The api api_key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            DashScopeAPIResponse: The response data.
        """
        return await super().cancel(task, api_key, workspace=workspace)

    @classmethod
    async def list(
        cls,
        start_time: str = None,
        end_time: str = None,
        model_name: str = None,
        api_key_id: str = None,
        region: str = None,
        status: str = None,
        page_no: int = 1,
        page_size: int = 10,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """List async tasks.

        Args:
            start_time (str, optional): The tasks start time,
                for example: 20230420000000. Defaults to None.
            end_time (str, optional): The tasks end time,
                for example: 20230420000000. Defaults to None.
            model_name (str, optional): The tasks model name. Defaults to None.
            api_key_id (str, optional): The tasks api-key-id. Defaults to None.
            region (str, optional): The service region,
                for example: cn-beijing. Defaults to None.
            status (str, optional): The status of tasks[PENDING,
                RUNNING, SUCCEEDED, FAILED, CANCELED]. Defaults to None.
            page_no (int, optional): The page number. Defaults to 1.
            page_size (int, optional): The page size. Defaults to 10.
            api_key (str, optional): The user api-key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            DashScopeAPIResponse: The response data.
        """
        return await super().list(
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
            **kwargs,
        )
