# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified async-task (Job) infrastructure (design doc M3 3.2).

Long-running services (images, videos, transcription, ...) share the
``X-DashScope-Async`` pattern: submit returns a ``task_id``, then the
task is fetched from ``/tasks/{task_id}`` until it reaches a terminal
status. ``Jobs`` / ``AsyncJobs`` wrap that pattern with exponential
backoff polling; resources mount them as ``jobs.create/fetch/wait``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dashscope.v2.errors import (
    DashScopeError,
    check_response,
    error_from_business_code,
)
from dashscope.v2.types.common import (
    ResponseBase,
    Usage,
    flatten_envelope,
    response_envelope,
    with_headers,
)

__all__ = [
    "AsyncJobs",
    "CANCELED",
    "FAILED",
    "Job",
    "JobList",
    "JobResult",
    "Jobs",
    "PENDING",
    "RUNNING",
    "SUCCEEDED",
    "UNKNOWN",
]

PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELED = "CANCELED"
UNKNOWN = "UNKNOWN"

# UNKNOWN counts as terminal even though it reads like an indeterminate
# state: the gateway answers ``GET /tasks/{task_id}`` with HTTP 200 and
# ``task_status: UNKNOWN`` for a task that does not exist (or is not
# visible to this api key/workspace), so it never resolves. Treating it as
# non-terminal would make ``wait()`` re-poll a permanently-dead id until
# the timeout — 24h by default — and then report it as a wait timeout,
# hiding the real cause. 1.x stops on it for the same reason
# (client/base_api.py lists TaskStatus.UNKNOWN alongside SUCCEEDED /
# FAILED / CANCELED); the fine-tune plane deliberately does not, and
# ``fine_tunes._TERMINAL`` mirrors that narrower 1.x rule.
_TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELED, UNKNOWN})

ASYNC_HEADER = {"X-DashScope-Async": "enable"}


def _image_choices(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Image parts of a choices-style output (multimodal-generation).

    Image models on the multimodal-generation protocol answer with
    chat-style ``choices`` instead of ``results``; lift every ``image``
    part so the typed views (``results`` / ``image_urls``) read the same
    for both output shapes.
    """
    results: List[Dict[str, Any]] = []
    for choice in output.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("image"):
                results.append({"url": part["image"]})
    return results


DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_MAX_INTERVAL = 8.0
# 1.x waits indefinitely; 2.0 caps the default at 24h so a stuck task
# cannot hang a process forever (recorded for the M7 migration guide).
DEFAULT_WAIT_TIMEOUT = 86400.0  # 24 hours


class JobResult(BaseModel):
    """One artifact a finished task produced.

    Image tasks answer with ``url``, transcription tasks with
    ``transcription_url``; anything else a service adds stays readable as an
    attribute.
    """

    model_config = ConfigDict(extra="allow")

    url: Optional[str] = None
    transcription_url: Optional[str] = None


class Job(ResponseBase):
    """One async task snapshot (submit or fetch result)."""

    task_id: str = ""
    status: str = Field(
        "",
        validation_alias=AliasChoices("status", "task_status"),
    )
    submitted_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("submitted_at", "submit_time"),
    )
    scheduled_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("scheduled_at", "scheduled_time"),
    )
    finished_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("finished_at", "end_time"),
    )
    # Raw task payload: result fields differ per service, so the dict stays.
    # ``results`` / ``image_urls`` / ``video_url`` / ``result_url`` are the
    # typed views over it.
    output: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None

    @property
    def job_id(self) -> str:
        """``FineTune`` spells the same identifier ``job_id``."""
        return self.task_id

    @property
    def succeeded(self) -> bool:
        return self.status == SUCCEEDED

    @property
    def finished(self) -> bool:
        return self.status in _TERMINAL

    @property
    def results(self) -> List[JobResult]:
        items = (self.output or {}).get("results") or []
        if not items:
            items = _image_choices(self.output or {})
        return [JobResult(**item) for item in items if isinstance(item, dict)]

    @property
    def image_urls(self) -> List[str]:
        return [result.url for result in self.results if result.url]

    @property
    def video_url(self) -> Optional[str]:
        return (self.output or {}).get("video_url")

    @property
    def result_url(self) -> Optional[str]:
        """The single artifact of a one-result task."""
        urls = self.image_urls
        if urls:
            return urls[0]
        output = self.output or {}
        return output.get("url") or self.video_url

    @classmethod
    def from_payload(cls, body: Dict[str, Any]) -> "Job":
        output = body.get("output")
        payload = flatten_envelope(body)
        payload["output"] = output if isinstance(output, dict) else None
        if not isinstance(payload.get("usage"), dict):
            payload.pop("usage", None)
        return cls(**payload)


class JobList(ResponseBase):
    """One page of async tasks, from ``jobs.list`` (1.x ``GET /tasks``).

    Each entry in ``tasks`` parses with the same aliases as a fetched
    :class:`Job`; a FAILED task inside the list is data, not an error.
    """

    page_no: int = 1
    page_size: int = 10
    total: int = 0
    tasks: List[Job] = []

    @classmethod
    def from_payload(cls, body: Dict[str, Any]) -> "JobList":
        # The list endpoint answers flat (no ``output`` wrapper), but
        # flatten_envelope also accepts an output-wrapped variant.
        data = flatten_envelope(body)
        tasks = data.pop("tasks", None) or []
        items = [task for task in tasks if isinstance(task, dict)]
        return cls(
            page_no=data.get("page_no", 1),
            page_size=data.get("page_size", 10),
            total=data.get("total", len(items)),
            tasks=items,
            **response_envelope(data),
        )


def _list_params(
    start_time: Optional[str],
    end_time: Optional[str],
    model_name: Optional[str],
    api_key_id: Optional[str],
    region: Optional[str],
    status: Optional[str],
    page_no: int,
    page_size: int,
) -> Dict[str, Any]:
    """Query params of 1.x ``GET /tasks``: pagination always, filters when set."""
    params: Dict[str, Any] = {"page_no": page_no, "page_size": page_size}
    for key, value in (
        ("start_time", start_time),
        ("end_time", end_time),
        ("model_name", model_name),
        ("api_key_id", api_key_id),
        ("region", region),
        ("status", status),
    ):
        if value is not None:
            params[key] = value
    return params


class Jobs:
    """Sync async-task helper for one service (images, videos, ...)."""

    def __init__(
        self,
        client,
        service: str,
        module: Optional[str] = None,
    ) -> None:
        self._client = client
        self._service = service
        # Functional-area tag for x-dashscope-sdk-client (see _ident);
        # the resource that mounts this helper passes its own module.
        self._module = module

    @property
    def submit_path(self) -> str:
        return f"/services/{self._service}"

    def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Submit a task; returns immediately with a PENDING/RUNNING Job."""
        body: Dict[str, Any] = {"model": model, "input": input}
        if parameters is not None:
            body["parameters"] = parameters
        if resources is not None:
            body["resources"] = resources
        submit_headers = dict(ASYNC_HEADER)
        if headers:
            submit_headers.update(headers)
        response = self._client.transport.request(
            "POST",
            self.submit_path,
            json=body,
            headers=submit_headers,
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return self._parse(response)

    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Fetch one task snapshot; raises on FAILED status."""
        response = self._client.transport.request(
            "GET",
            f"/tasks/{task_id}",
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return self._parse(response)

    def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Cancel a PENDING task (1.x: ``POST /tasks/{task_id}/cancel``)."""
        response = self._client.transport.request(
            "POST",
            f"/tasks/{task_id}/cancel",
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return self._parse(response)

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
        """List async tasks (1.x: ``GET /tasks`` with page_no/page_size)."""
        response = self._client.transport.request(
            "GET",
            "/tasks",
            params=_list_params(
                start_time,
                end_time,
                model_name,
                api_key_id,
                region,
                status,
                page_no,
                page_size,
            ),
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(JobList.from_payload(response.json()), response)

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
        """Poll with exponential backoff until the task finishes.

        With ``raise_on_error=False`` a terminal FAILED/CANCELED task (or
        a transport-level failure) comes back as the :class:`Job` envelope
        instead of raising; check ``job.status`` / ``job.is_error``.

        An ``UNKNOWN`` status is terminal too and is returned, not raised:
        the gateway reports it for a task id that does not exist, so
        polling again could never resolve it. Callers that care about the
        difference between "succeeded" and "no such task" should check
        ``job.succeeded`` rather than relying on ``wait()`` to raise.

        ``_sleep`` is a test seam — it replaces the pause between polls.
        """
        task_id = job.task_id if isinstance(job, Job) else job
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, Job)
            else self.fetch(task_id, api_key=api_key, workspace=workspace)
        )
        interval = poll_interval
        while not current.finished:
            # Compat mode (raise_on_error=False): a transport-level failure
            # comes back as an error envelope with no task status, which
            # would otherwise poll until the timeout.
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for task {task_id}",
                )
            sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = self.fetch(task_id, api_key=api_key, workspace=workspace)
        return self._check_canceled(current)

    def _parse(self, response: Any) -> Job:
        payload = response.json()
        job = with_headers(Job.from_payload(payload), response)
        if job.status == FAILED and self._client.raise_on_error:
            raise error_from_business_code(
                message=job.message or f"task {job.task_id} failed",
                code=job.code,
                body=payload,
                headers=response.headers,
            )
        return job

    def _check_canceled(self, job: Job) -> Job:
        if job.status == CANCELED and self._client.raise_on_error:
            raise DashScopeError(
                f"task {job.task_id} was canceled while waiting",
                request_id=job.request_id,
            )
        return job


class AsyncJobs:
    """Async async-task helper for one service (images, videos, ...)."""

    def __init__(
        self,
        client,
        service: str,
        module: Optional[str] = None,
    ) -> None:
        self._client = client
        self._service = service
        self._module = module

    @property
    def submit_path(self) -> str:
        return f"/services/{self._service}"

    async def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        body: Dict[str, Any] = {"model": model, "input": input}
        if parameters is not None:
            body["parameters"] = parameters
        if resources is not None:
            body["resources"] = resources
        submit_headers = dict(ASYNC_HEADER)
        if headers:
            submit_headers.update(headers)
        response = await self._client.transport.request(
            "POST",
            self.submit_path,
            json=body,
            headers=submit_headers,
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return self._parse(response)

    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        response = await self._client.transport.request(
            "GET",
            f"/tasks/{task_id}",
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return self._parse(response)

    async def cancel(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Cancel a PENDING task (1.x: ``POST /tasks/{task_id}/cancel``)."""
        response = await self._client.transport.request(
            "POST",
            f"/tasks/{task_id}/cancel",
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return self._parse(response)

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
        """List async tasks (1.x: ``GET /tasks`` with page_no/page_size)."""
        response = await self._client.transport.request(
            "GET",
            "/tasks",
            params=_list_params(
                start_time,
                end_time,
                model_name,
                api_key_id,
                region,
                status,
                page_no,
                page_size,
            ),
            timeout=timeout,
            sdk_module=self._module,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(JobList.from_payload(response.json()), response)

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
        """Async twin of :meth:`Jobs.wait`, same compat-mode contract."""
        task_id = job.task_id if isinstance(job, Job) else job
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, Job)
            else await self.fetch(
                task_id,
                api_key=api_key,
                workspace=workspace,
            )
        )
        interval = poll_interval
        while not current.finished:
            # Compat mode (raise_on_error=False): a transport-level failure
            # comes back as an error envelope with no task status, which
            # would otherwise poll until the timeout.
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for task {task_id}",
                )
            await sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = await self.fetch(
                task_id,
                api_key=api_key,
                workspace=workspace,
            )
        return self._check_canceled(current)

    def _parse(self, response: Any) -> Job:
        payload = response.json()
        job = with_headers(Job.from_payload(payload), response)
        if job.status == FAILED and self._client.raise_on_error:
            raise error_from_business_code(
                message=job.message or f"task {job.task_id} failed",
                code=job.code,
                body=payload,
                headers=response.headers,
            )
        return job

    def _check_canceled(self, job: Job) -> Job:
        if job.status == CANCELED and self._client.raise_on_error:
            raise DashScopeError(
                f"task {job.task_id} was canceled while waiting",
                request_id=job.request_id,
            )
        return job
