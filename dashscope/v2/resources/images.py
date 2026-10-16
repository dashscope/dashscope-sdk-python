# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.images resource (design doc M3 3.3).

Two wire protocols hide behind one resource, picked by model name:

* legacy text2image — ``/services/aigc/text2image/image-synthesis``,
  async tasks only, ``prompt`` input, ``results`` output. The endpoint
  serves wan2.5 and below (wan2.5-t2i-preview, wan2.2-t2i-*,
  wanx2.1-t2i-*, ...).
* multimodal-generation — one-shot sync calls hit
  ``/services/aigc/multimodal-generation/generation``, async tasks hit
  ``/services/aigc/image-generation/generation``; ``messages`` input,
  ``choices`` output. wan2.6-t2i / wan2.6-image only answer here.

The split follows the same convention as ``chat.completions`` routing
media messages to the multimodal endpoint on its own: the caller passes
a model, the resource picks the protocol. Either way ``generate``
returns the unified ``Job`` — ``results`` / ``image_urls`` read the
same for both output shapes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from dashscope.v2.errors import check_response
from dashscope.v2.resources._jobs import (
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    SUCCEEDED,
    AsyncJobs,
    Job,
    JobList,
    Jobs,
)
from dashscope.v2.resources._oss import (
    OSS_RESOLVE_HEADERS,
    upload_media_value,
)
from dashscope.v2.types.common import with_headers

__all__ = [
    "AsyncImages",
    "Images",
]

SERVICE = "aigc/text2image/image-synthesis"
MULTIMODAL_SERVICE = "aigc/multimodal-generation/generation"
TASK_SERVICE = "aigc/image-generation/generation"
# Functional-area tag for x-dashscope-sdk-client (1.x ImageSynthesis lives
# under the ``aigc`` package).
MODULE = "aigc"

# Models that only answer on the multimodal-generation protocol. Prefix
# match so ``-preview``-style variants route the same way.
MULTIMODAL_MODEL_PREFIXES = ("wan2.6-t2i", "wan2.6-image")


def _uses_multimodal_protocol(model: str) -> bool:
    return model.startswith(MULTIMODAL_MODEL_PREFIXES)


def _upload_inputs(
    client,
    model: str,
    image_urls: Optional[List[str]],
    extra_input: Optional[Dict[str, Any]],
    api_key: Optional[str],
    workspace: Optional[str],
) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]], bool]:
    """Upload local/oss reference images before the body is built."""
    has_upload = False
    if image_urls:
        image_urls, is_up = upload_media_value(
            client,
            list(image_urls),
            model,
            api_key,
            workspace,
        )
        has_upload = has_upload or is_up
    if extra_input:
        extra_input, is_up = upload_media_value(
            client,
            dict(extra_input),
            model,
            api_key,
            workspace,
        )
        has_upload = has_upload or is_up
    return image_urls, extra_input, has_upload


def _build_input(
    prompt: str,
    negative_prompt: Optional[str],
    extra_input: Optional[Dict[str, Any]],
    function: Optional[str] = None,
) -> Dict[str, Any]:
    input_data: Dict[str, Any] = {"prompt": prompt}
    if negative_prompt is not None:
        input_data["negative_prompt"] = negative_prompt
    if extra_input:
        input_data.update(extra_input)
    if function is not None:
        input_data["function"] = function
    return input_data


def _build_multimodal_input(
    prompt: str,
    image_urls: Optional[List[str]],
    extra_input: Optional[Dict[str, Any]],
    function: Optional[str] = None,
) -> Dict[str, Any]:
    """Messages-style input: text part plus one part per reference image."""
    content: List[Dict[str, Any]] = [{"text": prompt}]
    for url in image_urls or []:
        content.append({"image": url})
    input_data: Dict[str, Any] = {
        "messages": [{"role": "user", "content": content}],
    }
    if extra_input:
        input_data.update(extra_input)
    if function is not None:
        input_data["function"] = function
    return input_data


def _normalize_size(size: str) -> str:
    """Accept ``1024x1024`` as well as the wire form ``1024*1024``."""
    return size.replace("x", "*").replace("X", "*")


def _merge_named_parameters(
    parameters: Optional[Dict[str, Any]],
    named: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Merge named kwargs into ``parameters``; the named kwarg wins."""
    merged = dict(parameters or {})
    for key, value in named.items():
        if value is not None:
            merged[key] = value
    return merged or None


def _named_image_parameters(
    n: Optional[int],
    size: Optional[str],
    seed: Optional[int],
    style: Optional[str],
    prompt_extend: Optional[bool],
    watermark: Optional[bool],
) -> Dict[str, Any]:
    """Named kwargs shared by both image protocols (1.x wire spellings)."""
    named: Dict[str, Any] = {
        "n": n,
        "seed": seed,
        "style": style,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if size is not None:
        named["size"] = _normalize_size(size)
    return named


def _build_parameters(
    n: Optional[int],
    size: Optional[str],
    parameters: Optional[Dict[str, Any]],
    *,
    seed: Optional[int] = None,
    style: Optional[str] = None,
    prompt_extend: Optional[bool] = None,
    watermark: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    return _merge_named_parameters(
        parameters,
        _named_image_parameters(
            n,
            size,
            seed,
            style,
            prompt_extend,
            watermark,
        ),
    )


def _build_multimodal_parameters(
    negative_prompt: Optional[str],
    n: Optional[int],
    size: Optional[str],
    parameters: Optional[Dict[str, Any]],
    *,
    seed: Optional[int] = None,
    style: Optional[str] = None,
    prompt_extend: Optional[bool] = None,
    watermark: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """The multimodal protocol takes ``negative_prompt`` as a parameter."""
    named = _named_image_parameters(
        n,
        size,
        seed,
        style,
        prompt_extend,
        watermark,
    )
    named["negative_prompt"] = negative_prompt
    return _merge_named_parameters(parameters, named)


def _build_multimodal_body(
    model: str,
    prompt: str,
    negative_prompt: Optional[str],
    n: Optional[int],
    size: Optional[str],
    image_urls: Optional[List[str]],
    extra_input: Optional[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]],
    *,
    seed: Optional[int] = None,
    style: Optional[str] = None,
    prompt_extend: Optional[bool] = None,
    watermark: Optional[bool] = None,
    function: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "input": _build_multimodal_input(
            prompt,
            image_urls,
            extra_input,
            function,
        ),
    }
    merged = _build_multimodal_parameters(
        negative_prompt,
        n,
        size,
        parameters,
        seed=seed,
        style=style,
        prompt_extend=prompt_extend,
        watermark=watermark,
    )
    if merged:
        body["parameters"] = merged
    return body


def _parse_sync_job(payload: Dict[str, Any]) -> Job:
    """Wrap a one-shot multimodal answer in the unified Job contract.

    A sync response carries no task fields; mark it finished so
    ``succeeded`` / ``finished`` read right. A business error body
    (``raise_on_error=False``) keeps its code instead.
    """
    output = payload.get("output")
    if isinstance(output, dict) and not payload.get("code"):
        output.setdefault("task_status", SUCCEEDED)
    return Job.from_payload(payload)


class ImageJobs:
    """Async task facade: create/fetch/wait over the image services.

    ``create`` submits legacy models to the text2image endpoint and
    multimodal-protocol models to the image-generation task endpoint;
    ``fetch`` / ``wait`` poll ``/tasks/{task_id}`` either way.
    """

    def __init__(self, client) -> None:
        self._client = client
        self._synthesis = Jobs(client, SERVICE, MODULE)
        self._generation = Jobs(client, TASK_SERVICE, MODULE)

    def create(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        style: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        image_urls, extra_input, has_upload = _upload_inputs(
            self._client,
            model,
            image_urls,
            extra_input,
            api_key,
            workspace,
        )
        resolve = OSS_RESOLVE_HEADERS if has_upload else None
        if _uses_multimodal_protocol(model):
            return self._generation.create(
                model=model,
                input=_build_multimodal_input(
                    prompt,
                    image_urls,
                    extra_input,
                    function,
                ),
                parameters=_build_multimodal_parameters(
                    negative_prompt,
                    n,
                    size,
                    parameters,
                    seed=seed,
                    style=style,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
                headers=resolve,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
            )
        return self._synthesis.create(
            model=model,
            input=_build_input(prompt, negative_prompt, extra_input, function),
            parameters=_build_parameters(
                n,
                size,
                parameters,
                seed=seed,
                style=style,
                prompt_extend=prompt_extend,
                watermark=watermark,
            ),
            headers=resolve,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return self._synthesis.fetch(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
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
        return self._synthesis.cancel(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
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
        return self._synthesis.list(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
            api_key_id=api_key_id,
            region=region,
            status=status,
            page_no=page_no,
            page_size=page_size,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
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
        return self._synthesis.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            _sleep=_sleep,
        )


class Images:
    """Text-to-image: one-shot generate + jobs.create/fetch/wait."""

    def __init__(self, client) -> None:
        self._client = client
        self.jobs = ImageJobs(client)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        style: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        submit_timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Submit and wait; returns the finished Job (``image_urls``).

        ``timeout`` is the wait budget — the same ``timeout`` as
        ``jobs.wait``. ``submit_timeout`` caps only the submit request.

        Multimodal-protocol models (wan2.6-t2i / wan2.6-image) answer
        synchronously: the call is a single request capped by
        ``submit_timeout`` and the polling knobs do not apply.
        ``image_urls`` carries reference images for wan2.6-image.
        """
        if _uses_multimodal_protocol(model):
            image_urls, extra_input, has_upload = _upload_inputs(
                self._client,
                model,
                image_urls,
                extra_input,
                api_key,
                workspace,
            )
            response = self._client.transport.request(
                "POST",
                f"/services/{MULTIMODAL_SERVICE}",
                json=_build_multimodal_body(
                    model,
                    prompt,
                    negative_prompt,
                    n,
                    size,
                    image_urls,
                    extra_input,
                    parameters,
                    seed=seed,
                    style=style,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    function=function,
                ),
                headers=OSS_RESOLVE_HEADERS if has_upload else None,
                timeout=submit_timeout,
                sdk_module=MODULE,
                api_key=api_key,
                workspace=workspace,
            )
            check_response(
                response,
                raise_on_error=self._client.raise_on_error,
            )
            return with_headers(
                _parse_sync_job(response.json()),
                response,
            )
        job = self.jobs.create(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            n=n,
            size=size,
            image_urls=image_urls,
            extra_input=extra_input,
            function=function,
            seed=seed,
            style=style,
            prompt_extend=prompt_extend,
            watermark=watermark,
            parameters=parameters,
            timeout=submit_timeout,
            api_key=api_key,
            workspace=workspace,
        )
        return self.jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )


class AsyncImageJobs:
    """Async counterpart of :class:`ImageJobs`."""

    def __init__(self, client) -> None:
        self._client = client
        self._synthesis = AsyncJobs(client, SERVICE, MODULE)
        self._generation = AsyncJobs(client, TASK_SERVICE, MODULE)

    async def create(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        style: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        # The upload path is blocking (requests + getPolicy); run it off the
        # event loop so it never stalls other tasks.
        image_urls, extra_input, has_upload = await asyncio.to_thread(
            _upload_inputs,
            self._client,
            model,
            image_urls,
            extra_input,
            api_key,
            workspace,
        )
        resolve = OSS_RESOLVE_HEADERS if has_upload else None
        if _uses_multimodal_protocol(model):
            return await self._generation.create(
                model=model,
                input=_build_multimodal_input(
                    prompt,
                    image_urls,
                    extra_input,
                    function,
                ),
                parameters=_build_multimodal_parameters(
                    negative_prompt,
                    n,
                    size,
                    parameters,
                    seed=seed,
                    style=style,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
                headers=resolve,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
            )
        return await self._synthesis.create(
            model=model,
            input=_build_input(prompt, negative_prompt, extra_input, function),
            parameters=_build_parameters(
                n,
                size,
                parameters,
                seed=seed,
                style=style,
                prompt_extend=prompt_extend,
                watermark=watermark,
            ),
            headers=resolve,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return await self._synthesis.fetch(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
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
        return await self._synthesis.cancel(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
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
        return await self._synthesis.list(
            start_time=start_time,
            end_time=end_time,
            model_name=model_name,
            api_key_id=api_key_id,
            region=region,
            status=status,
            page_no=page_no,
            page_size=page_size,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
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
        return await self._synthesis.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            _sleep=_sleep,
        )


class AsyncImages:
    """Async counterpart of :class:`Images`."""

    def __init__(self, client) -> None:
        self._client = client
        self.jobs = AsyncImageJobs(client)

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        style: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        submit_timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """See :meth:`Images.generate`."""
        if _uses_multimodal_protocol(model):
            # The upload path is blocking (requests + getPolicy); run it off
            # the event loop so it never stalls other tasks.
            image_urls, extra_input, has_upload = await asyncio.to_thread(
                _upload_inputs,
                self._client,
                model,
                image_urls,
                extra_input,
                api_key,
                workspace,
            )
            response = await self._client.transport.request(
                "POST",
                f"/services/{MULTIMODAL_SERVICE}",
                json=_build_multimodal_body(
                    model,
                    prompt,
                    negative_prompt,
                    n,
                    size,
                    image_urls,
                    extra_input,
                    parameters,
                    seed=seed,
                    style=style,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    function=function,
                ),
                headers=OSS_RESOLVE_HEADERS if has_upload else None,
                timeout=submit_timeout,
                sdk_module=MODULE,
                api_key=api_key,
                workspace=workspace,
            )
            check_response(
                response,
                raise_on_error=self._client.raise_on_error,
            )
            return with_headers(
                _parse_sync_job(response.json()),
                response,
            )
        job = await self.jobs.create(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            n=n,
            size=size,
            image_urls=image_urls,
            extra_input=extra_input,
            function=function,
            seed=seed,
            style=style,
            prompt_extend=prompt_extend,
            watermark=watermark,
            parameters=parameters,
            timeout=submit_timeout,
            api_key=api_key,
            workspace=workspace,
        )
        return await self.jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
