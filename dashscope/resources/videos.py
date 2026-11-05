# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.videos resource (design doc M3 3.3).

One service, two input protocols, picked by model name:

* legacy — ``img_url`` first-frame input (the public name is
  ``image_url``), ``prompt``-only works too. Serves wan2.6 and below
  (wanx2.1-t2v-*, wan2.6-i2v, ...).
* media — ``media`` input: a list of typed assets, one per kind
  (``first_frame`` / ``last_frame`` / ``driving_audio`` /
  ``first_clip``). wan2.7-i2v only answers here; ``image_url`` is
  accepted as shorthand for a lone ``first_frame`` asset.

Both protocols submit to the same video-synthesis endpoint and answer
the same async-task envelope, so ``generate`` returns the unified
``Job`` — ``video_url`` reads the same either way.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    overload,
)

from dashscope.resources._blocking import run_blocking
from dashscope.resources._jobs import (
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    AsyncJobs,
    Job,
    JobList,
    Jobs,
)
from dashscope.resources._oss import (
    OSS_RESOLVE_HEADERS,
    upload_media_value,
)

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = [
    "AsyncVideos",
    "Videos",
]

SERVICE = "aigc/video-generation/video-synthesis"
# Functional-area tag for x-dashscope-sdk-client (1.x VideoSynthesis lives
# under the ``aigc`` package).
MODULE = "aigc"

# Models that only answer on the media-based input protocol. Prefix
# match so dated variants (``wan2.7-i2v-2026-04-25``) route the same way.
MEDIA_MODEL_PREFIXES = ("wan2.7-i2v",)


def _uses_media_protocol(model: str) -> bool:
    return model.startswith(MEDIA_MODEL_PREFIXES)


def _upload_inputs(
    client,
    model: str,
    image_url: Optional[str],
    media: Optional[List[Dict[str, Any]]],
    extra_input: Optional[Dict[str, Any]],
    api_key: Optional[str],
    workspace: Optional[str],
) -> Tuple[
    Optional[str],
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, Any]],
    bool,
]:
    """Upload local/oss first-frame and media assets before the body."""
    has_upload = False
    if image_url is not None:
        image_url, is_up = upload_media_value(
            client,
            image_url,
            model,
            api_key,
            workspace,
        )
        has_upload = has_upload or is_up
    if media:
        media, is_up = upload_media_value(
            client,
            [dict(item) for item in media],
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
    return image_url, media, extra_input, has_upload


def _build_input(
    model: str,
    prompt: Optional[str],
    image_url: Optional[str],
    media: Optional[List[Dict[str, Any]]],
    extra_input: Optional[Dict[str, Any]],
    function: Optional[str] = None,
) -> Dict[str, Any]:
    input_data: Dict[str, Any] = {}
    if prompt is not None:
        input_data["prompt"] = prompt
    if _uses_media_protocol(model):
        items = [dict(item) for item in media or []]
        if image_url is not None:
            # ``image_url`` is shorthand for a lone first_frame asset.
            if any(item.get("type") == "first_frame" for item in items):
                raise ValueError(
                    "image_url duplicates a first_frame media item",
                )
            items.insert(0, {"type": "first_frame", "url": image_url})
        if not items:
            raise ValueError(
                f"videos.generate requires media or image_url with {model}",
            )
        input_data["media"] = items
    else:
        if media:
            input_data["media"] = [dict(item) for item in media]
        if image_url is not None:
            # Wire spelling; the public name is ``image_url``.
            input_data["img_url"] = image_url
    if extra_input:
        input_data.update(extra_input)
    if function is not None:
        input_data["function"] = function
    if not input_data:
        raise ValueError("videos.generate requires prompt or image_url")
    return input_data


def _build_parameters(
    *,
    seed: Optional[int],
    size: Optional[str],
    duration: Optional[int],
    resolution: Optional[str],
    ratio: Optional[str],
    prompt_extend: Optional[bool],
    watermark: Optional[bool],
    parameters: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge the 1.x video kwargs into ``parameters``; the named kwarg wins."""
    merged = dict(parameters or {})
    for key, value in (
        ("seed", seed),
        ("size", size),
        ("duration", duration),
        ("resolution", resolution),
        ("ratio", ratio),
        ("prompt_extend", prompt_extend),
        ("watermark", watermark),
    ):
        if value is not None:
            merged[key] = value
    return merged or None


class VideoJobs:
    """Async task facade: create/fetch/wait over the video service."""

    def __init__(self, client) -> None:
        self._client = client
        self._jobs = Jobs(client, SERVICE, MODULE)

    @overload
    def create(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Job:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def create(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Job, "APIResponse"]:
        """Submit a video task; returns a PENDING/RUNNING Job.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying submit call before any error check.
        """
        image_url, media, extra_input, has_upload = _upload_inputs(
            self._client,
            model,
            image_url,
            media,
            extra_input,
            api_key,
            workspace,
        )
        return self._jobs.create(
            model=model,
            input=_build_input(
                model,
                prompt,
                image_url,
                media,
                extra_input,
                function,
            ),
            parameters=_build_parameters(
                seed=seed,
                size=size,
                duration=duration,
                resolution=resolution,
                ratio=ratio,
                prompt_extend=prompt_extend,
                watermark=watermark,
                parameters=parameters,
            ),
            headers=OSS_RESOLVE_HEADERS if has_upload else None,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
        )

    @overload
    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Job:
        ...

    @overload
    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Job, "APIResponse"]:
        """Fetch one video task snapshot.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying fetch call before any error check.
        """
        return self._jobs.fetch(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
        )

    @overload
    def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Job:
        ...

    @overload
    def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Job, "APIResponse"]:
        """Cancel a PENDING task (``POST /tasks/{task_id}/cancel``).

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying cancel call before any error check.
        """
        return self._jobs.cancel(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
        )

    @overload
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobList:
        ...

    @overload
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobList, "APIResponse"]:
        """List async tasks (``GET /tasks``).

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying list call before any error check.
        """
        return self._jobs.list(
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
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> Job:
        return self._jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            _sleep=_sleep,
        )


class Videos:
    """Text/image-to-video: one-shot generate + jobs.create/fetch/wait."""

    def __init__(self, client) -> None:
        self._client = client
        self.jobs = VideoJobs(client)

    def generate(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        submit_timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Job:
        """Submit and wait; returns the finished Job (output.video_url).

        ``timeout`` is the wait budget — the same ``timeout`` as
        ``jobs.wait``. ``submit_timeout`` caps only the submit request.

        Media-protocol models (wan2.7-i2v) require ``media`` — one
        item per asset type (``first_frame`` / ``last_frame`` /
        ``driving_audio`` / ``first_clip``); ``image_url`` still works
        as a first-frame shorthand. Older models take ``image_url``
        (wire: ``img_url``) or a bare ``prompt``.
        """
        job = self.jobs.create(
            model=model,
            prompt=prompt,
            image_url=image_url,
            media=media,
            extra_input=extra_input,
            function=function,
            seed=seed,
            size=size,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            parameters=parameters,
            timeout=submit_timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )
        return self.jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )


class AsyncVideoJobs:
    def __init__(self, client) -> None:
        self._client = client
        self._jobs = AsyncJobs(client, SERVICE, MODULE)

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Job:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def create(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Job, "APIResponse"]:
        """Submit a video task; returns a PENDING/RUNNING Job.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying submit call before any error check.
        """
        # The upload path is blocking (file post + getPolicy), so it runs in
        # a worker thread -- over a transport bridged back to this loop,
        # which is what lets it serve the policy fetch.
        image_url, media, extra_input, has_upload = await run_blocking(
            self._client,
            _upload_inputs,
            model,
            image_url,
            media,
            extra_input,
            api_key,
            workspace,
        )
        return await self._jobs.create(
            model=model,
            input=_build_input(
                model,
                prompt,
                image_url,
                media,
                extra_input,
                function,
            ),
            parameters=_build_parameters(
                seed=seed,
                size=size,
                duration=duration,
                resolution=resolution,
                ratio=ratio,
                prompt_extend=prompt_extend,
                watermark=watermark,
                parameters=parameters,
            ),
            headers=OSS_RESOLVE_HEADERS if has_upload else None,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
        )

    @overload
    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Job:
        ...

    @overload
    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Job, "APIResponse"]:
        """Fetch one video task snapshot.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying fetch call before any error check.
        """
        return await self._jobs.fetch(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
        )

    @overload
    async def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Job:
        ...

    @overload
    async def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Job, "APIResponse"]:
        """Cancel a PENDING task (``POST /tasks/{task_id}/cancel``).

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying cancel call before any error check.
        """
        return await self._jobs.cancel(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
        )

    @overload
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobList:
        ...

    @overload
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobList, "APIResponse"]:
        """List async tasks (``GET /tasks``).

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        of the underlying list call before any error check.
        """
        return await self._jobs.list(
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
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            raw=raw,
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        _sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> Job:
        return await self._jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            _sleep=_sleep,
        )


class AsyncVideos:
    def __init__(self, client) -> None:
        self._client = client
        self.jobs = AsyncVideoJobs(client)

    async def generate(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        media: Optional[List[Dict[str, Any]]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        function: Optional[str] = None,
        seed: Optional[int] = None,
        size: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        prompt_extend: Optional[bool] = None,
        watermark: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        submit_timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Job:
        """See :meth:`Videos.generate`."""
        job = await self.jobs.create(
            model=model,
            prompt=prompt,
            image_url=image_url,
            media=media,
            extra_input=extra_input,
            function=function,
            seed=seed,
            size=size,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            parameters=parameters,
            timeout=submit_timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )
        return await self.jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )
