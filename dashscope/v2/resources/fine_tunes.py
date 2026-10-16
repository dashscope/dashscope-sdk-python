# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.fine_tunes resource (design doc M3 3.6).

Wire contract follows the 1.x ``FineTunes`` REST surface, except that
``create`` names the produced model with ``model_name`` (1.x sent the
response-only ``finetuned_output``, which current servers reject):
``POST /fine-tunes``, ``GET /fine-tunes``, ``GET /fine-tunes/{id}``,
``POST /fine-tunes/{id}/cancel``, ``DELETE /fine-tunes/{id}``,
``GET /fine-tunes/{id}/logs``, ``GET /fine-tunes/{id}/stream`` (SSE job
events).
"""

from __future__ import annotations

import asyncio
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from dashscope.v2.errors import (
    DashScopeError,
    check_response,
    error_from_business_code,
)
from dashscope.v2.resources._jobs import (
    CANCELED,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    FAILED,
    SUCCEEDED,
)
from dashscope.v2.resources._sse import (
    _aiter_sse,
    _iter_sse,
    _stream_error,
    _try_json,
)
from dashscope.v2.types.common import (
    JobStatus,
    ResponseBase,
    Usage,
    flatten_envelope,
    response_envelope,
    with_headers,
)

_BASE_PATH = "/fine-tunes"

MODULE = "finetune"

__all__ = [
    "AsyncFineTunes",
    "FineTune",
    "FineTuneEvent",
    "FineTuneList",
    "FineTunes",
    "JobLogs",
]

_TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELED})


class FineTune(ResponseBase):
    """One fine-tune job.

    The same type comes back from ``create`` / ``get`` and is nested in
    ``FineTuneList.jobs``, so there is one shape to learn. The server reports
    a failure ``code`` per job, so the envelope is meaningful in both roles:
    a failed job in a list flags itself through ``is_error`` as well.
    """

    job_id: Optional[str] = None
    job_name: Optional[str] = None
    status: Optional[str] = None
    model: Optional[str] = None
    base_model: Optional[str] = None
    finetuned_model: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("finetuned_model", "finetuned_output"),
    )
    training_file_ids: List[str] = []
    validation_file_ids: List[str] = []
    hyper_parameters: Dict[str, Any] = {}
    training_type: Optional[str] = None
    created_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("created_at", "create_time"),
    )
    finished_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("finished_at", "end_time"),
    )
    user_identity: Optional[str] = None
    modifier: Optional[str] = None
    creator: Optional[str] = None
    group: Optional[str] = None
    usage: Optional[Usage] = None

    @property
    def task_id(self) -> Optional[str]:
        """``Job`` spells the same identifier ``task_id``."""
        return self.job_id

    @property
    def succeeded(self) -> bool:
        return self.status == SUCCEEDED

    @property
    def finished(self) -> bool:
        return self.status in _TERMINAL

    @field_validator("usage", mode="before")
    @classmethod
    def _usage_from_count(cls, value: Any) -> Any:
        # Fine-tune jobs may report usage as a bare token count rather than
        # the usual object.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"total_tokens": int(value)}
        return value


class FineTuneList(ResponseBase):
    page_no: int = 1
    page_size: int = 10
    total_count: int = 0
    jobs: List[FineTune] = []


class JobLogs(ResponseBase):
    """Training logs of one job, from ``fine_tunes.logs``."""

    job_id: Optional[str] = None
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _content_from_lines(cls, value: Any) -> Any:
        # The server may return logs either as a single string or as an
        # array of lines; normalize the array form into newline-joined text
        # so the public ``content`` contract stays a plain string.
        if isinstance(value, (list, tuple)):
            return "\n".join(
                "" if item is None else str(item) for item in value
            )
        return value


class FineTuneEvent(BaseModel):
    """One SSE frame from ``/fine-tunes/{id}/stream``."""

    model_config = ConfigDict(extra="allow")

    output: Union[Dict[str, Any], str, None] = None
    is_error: bool = False


def _parse_job(body: Dict[str, Any]) -> FineTune:
    return FineTune(**flatten_envelope(body))


def _parse_list(body: Dict[str, Any]) -> FineTuneList:
    data = flatten_envelope(body)
    jobs = data.get("jobs") or []
    return FineTuneList(
        page_no=data.get("page_no", 1),
        page_size=data.get("page_size", 10),
        total_count=data.get("total", len(jobs)),
        jobs=jobs,
        **response_envelope(data),
    )


def _parse_status(body: Dict[str, Any]) -> JobStatus:
    return JobStatus(**flatten_envelope(body))


def _parse_logs(job_id: str, body: Dict[str, Any]) -> JobLogs:
    data = flatten_envelope(body)
    return JobLogs(
        job_id=job_id,
        content=data.get("logs") or "",
        **response_envelope(data),
    )


def _job_id(job: Union[FineTune, str]) -> str:
    job_id = job.job_id if isinstance(job, FineTune) else job
    if not job_id:
        raise DashScopeError("fine-tune job id required")
    return job_id


def _build_create_body(
    model: str,
    training_file_ids: Union[List[str], str],
    validation_file_ids: Union[List[str], str, None],
    mode: Optional[str],
    hyper_parameters: Optional[Dict[str, Any]],
    finetuned_model: Optional[str],
) -> Dict[str, Any]:
    if isinstance(training_file_ids, str):
        training_file_ids = [training_file_ids]
    if validation_file_ids and isinstance(validation_file_ids, str):
        validation_file_ids = [validation_file_ids]
    body: Dict[str, Any] = {
        "model": model,
        "training_file_ids": training_file_ids,
        "validation_file_ids": validation_file_ids,
        "hyper_parameters": hyper_parameters or {},
    }
    if mode is not None:
        body["training_type"] = mode
    if finetuned_model is not None:
        # The server names the produced model via ``model_name``; the
        # response-only ``finetuned_output`` cannot be specified on create.
        body["model_name"] = finetuned_model
    return body


class FineTunes:
    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        training_file_ids: Union[List[str], str],
        validation_file_ids: Union[List[str], str, None] = None,
        mode: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        finetuned_model: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FineTune:
        body = _build_create_body(
            model,
            training_file_ids,
            validation_file_ids,
            mode,
            hyper_parameters,
            finetuned_model,
        )
        response = self._client.transport.request(
            "POST",
            _BASE_PATH,
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_job(response.json()), response)

    def get(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FineTune:
        response = self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_job(response.json()), response)

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FineTuneList:
        response = self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_list(response.json()), response)

    def cancel(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        response = self._client.transport.request(
            "POST",
            f"{_BASE_PATH}/{job_id}/cancel",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_status(response.json()), response)

    def delete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        response = self._client.transport.request(
            "DELETE",
            f"{_BASE_PATH}/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_status(response.json()), response)

    def logs(
        self,
        job_id: str,
        *,
        offset: int = 1,
        lines: int = 1000,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobLogs:
        response = self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{job_id}/logs",
            # The wire parameter is singular "line".
            params={"offset": offset, "line": lines},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_logs(job_id, response.json()), response)

    def stream_events(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Iterator[FineTuneEvent]:
        response = self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{job_id}/stream",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        try:
            for is_error, status_code, data in _iter_sse(response):
                if is_error:
                    if self._client.raise_on_error:
                        raise _stream_error(status_code, data)
                    yield FineTuneEvent(output=_try_json(data), is_error=True)
                    return
                yield FineTuneEvent(output=_try_json(data))
        finally:
            response.close()

    def wait(
        self,
        job: Union[FineTune, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> FineTune:
        """Poll until the job reaches a terminal status.

        With ``raise_on_error=False`` a transport-level failure comes
        back as the :class:`FineTune` envelope instead of raising.
        ``_sleep`` is a test seam — it replaces the pause between polls.
        """
        job_id = _job_id(job)
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, FineTune)
            else self.get(job_id, api_key=api_key, workspace=workspace)
        )
        interval = poll_interval
        while not current.finished:
            # Compat mode (raise_on_error=False): a transport-level failure
            # comes back as an error envelope with no job status, which
            # would otherwise poll until the timeout.
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for"
                    f" fine-tune job {job_id}",
                )
            sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = self.get(job_id, api_key=api_key, workspace=workspace)
        return self._check_terminal(current, job_id)

    def _check_terminal(self, job: FineTune, job_id: str) -> FineTune:
        if job.status == FAILED and self._client.raise_on_error:
            raise error_from_business_code(
                message=job.message or f"fine-tune job {job_id} failed",
                code=job.code,
                # ``headers`` is transport metadata, not wire body.
                body=job.model_dump(exclude={"headers"}),
                headers=job.headers,
            )
        if job.status == CANCELED and self._client.raise_on_error:
            raise DashScopeError(
                f"fine-tune job {job_id} was canceled while waiting",
                request_id=job.request_id,
            )
        return job


class AsyncFineTunes:
    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        training_file_ids: Union[List[str], str],
        validation_file_ids: Union[List[str], str, None] = None,
        mode: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        finetuned_model: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FineTune:
        body = _build_create_body(
            model,
            training_file_ids,
            validation_file_ids,
            mode,
            hyper_parameters,
            finetuned_model,
        )
        response = await self._client.transport.request(
            "POST",
            _BASE_PATH,
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_job(response.json()), response)

    async def get(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FineTune:
        response = await self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_job(response.json()), response)

    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FineTuneList:
        response = await self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_list(response.json()), response)

    async def cancel(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        response = await self._client.transport.request(
            "POST",
            f"{_BASE_PATH}/{job_id}/cancel",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_status(response.json()), response)

    async def delete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        response = await self._client.transport.request(
            "DELETE",
            f"{_BASE_PATH}/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_status(response.json()), response)

    async def logs(
        self,
        job_id: str,
        *,
        offset: int = 1,
        lines: int = 1000,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobLogs:
        response = await self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{job_id}/logs",
            # The wire parameter is singular "line".
            params={"offset": offset, "line": lines},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_logs(job_id, response.json()), response)

    async def stream_events(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> AsyncIterator[FineTuneEvent]:
        response = await self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{job_id}/stream",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        try:
            async for is_error, status_code, data in _aiter_sse(response):
                if is_error:
                    if self._client.raise_on_error:
                        raise _stream_error(status_code, data)
                    yield FineTuneEvent(
                        output=_try_json(data),
                        is_error=True,
                    )
                    return
                yield FineTuneEvent(output=_try_json(data))
        finally:
            await response.aclose()

    async def wait(
        self,
        job: Union[FineTune, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        _sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> FineTune:
        job_id = _job_id(job)
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, FineTune)
            else await self.get(
                job_id,
                api_key=api_key,
                workspace=workspace,
            )
        )
        interval = poll_interval
        while not current.finished:
            # Compat mode (raise_on_error=False): a transport-level failure
            # comes back as an error envelope with no job status, which
            # would otherwise poll until the timeout.
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for"
                    f" fine-tune job {job_id}",
                )
            await sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = await self.get(
                job_id,
                api_key=api_key,
                workspace=workspace,
            )
        return self._check_terminal(current, job_id)

    def _check_terminal(self, job: FineTune, job_id: str) -> FineTune:
        if job.status == FAILED and self._client.raise_on_error:
            raise error_from_business_code(
                message=job.message or f"fine-tune job {job_id} failed",
                code=job.code,
                # ``headers`` is transport metadata, not wire body.
                body=job.model_dump(exclude={"headers"}),
                headers=job.headers,
            )
        if job.status == CANCELED and self._client.raise_on_error:
            raise DashScopeError(
                f"fine-tune job {job_id} was canceled while waiting",
                request_id=job.request_id,
            )
        return job
