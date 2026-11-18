# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.fine_tunes.compress -- model compression (quantization) jobs.

A REST job plane nested under fine-tunes, mirroring the parent's
create/get/list/logs/cancel/delete/wait surface rather than the
``/services`` + ``/tasks`` async pattern: the wire contract is
``/api/v1/fine-tunes/compress`` with ``templates`` and ``jobs`` beneath it.

Compression is quantization-only and currently Beijing-region only; the
base model must be a full-parameter fine-tune (LoRA / already-quantized
models are not supported). The produced ``quantized_output`` model name
feeds ``deployments``. Auth is the same API-Key Bearer as the rest of the
core Client, so this lives here rather than in a RAM ManagementClient.
"""

from __future__ import annotations

import asyncio
import re
import time
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
    Union,
    overload,
)

from pydantic import BaseModel, ConfigDict

from dashscope.errors import (
    DashScopeError,
    error_from_business_code,
)
from dashscope.resources._jobs import (
    CANCELED,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    FAILED,
    SUCCEEDED,
)
from dashscope.resources.fine_tunes import JobLogs
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    response_envelope,
    with_headers,
)
from dashscope.types.pagination import BasePage

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_BASE_PATH = "/fine-tunes/compress"

MODULE = "finetune"

__all__ = [
    "AsyncCompress",
    "AsyncCompressJobs",
    "AsyncCompressTemplates",
    "Compress",
    "CompressCustomModel",
    "CompressError",
    "CompressHyperParameter",
    "CompressJob",
    "CompressJobList",
    "CompressJobs",
    "CompressTemplate",
    "CompressTemplateList",
    "CompressTemplates",
]

_TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELED})

_SUFFIX_RE = re.compile(r"[a-z0-9]{1,8}")


class CompressError(BaseModel):
    """The ``error`` object a failed compress job carries."""

    model_config = ConfigDict(extra="allow")

    code: Optional[str] = None
    message: Optional[str] = None


class CompressJob(ResponseBase):
    """One compression job.

    The same type comes back from ``create`` / ``get`` and is nested in
    ``CompressJobList.jobs``. A failed job reports its own ``error`` object
    alongside the response envelope, so both stay readable.
    """

    job_id: Optional[str] = None
    job_name: Optional[str] = None
    job_description: Optional[str] = None
    status: Optional[str] = None
    model: Optional[str] = None
    base_model: Optional[str] = None
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    template_description: Optional[str] = None
    training_type: Optional[str] = None
    compress_type: Optional[str] = None
    hyper_parameters: Dict[str, Any] = {}
    custom_calibration_file_ids: List[str] = []
    quantized_output: Optional[str] = None
    create_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[CompressError] = None
    group: Optional[str] = None
    # GPU seconds (Integer on the wire), present on SUCCEEDED / CANCELED.
    usage: Optional[int] = None

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


class CompressJobList(BasePage[CompressJob]):
    """One page of compression jobs, from ``compress.jobs.list``.

    ``page_no`` / ``page_size`` / ``total_count`` come from
    :class:`~dashscope.types.pagination.BasePage`; iterate the page directly
    or cross pages with ``iter_pages()`` / ``next_page()``.
    """

    jobs: List[CompressJob] = []

    @property
    def items(self) -> List[CompressJob]:
        return self.jobs

    @property
    def total(self) -> int:
        """Alias of ``total_count``, keeping the wire field name readable."""
        return self.total_count


class CompressHyperParameter(BaseModel):
    """One tunable hyper-parameter descriptor a template advertises.

    ``type=string`` carries ``support_values`` (an enum); ``type=number``
    carries ``data_range`` + ``step``. The ``name`` is the key to use when
    overriding ``hyper_parameters`` on ``jobs.create``.
    """

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    type: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    defaultValue: Optional[str] = None
    recommend_value: Optional[str] = None
    required: Optional[bool] = None
    support_values: List[str] = []
    data_range: List[str] = []
    step: Optional[int] = None


class CompressTemplate(BaseModel):
    """One compression template bound to a custom model.

    Nested under ``CompressCustomModel.templates``; the wire field is
    ``description`` (not ``template_description``) and ``hyper_parameters``
    is an array of descriptors, empty when nothing is tunable.
    """

    model_config = ConfigDict(extra="allow")

    template_id: Optional[str] = None
    template_name: Optional[str] = None
    description: Optional[str] = None
    compress_type: Optional[str] = None
    hyper_parameters: List[CompressHyperParameter] = []


class CompressCustomModel(BaseModel):
    """One quantizable custom fine-tune and the templates it supports."""

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    model_name: Optional[str] = None
    base_model: Optional[str] = None
    templates: List[CompressTemplate] = []


class CompressTemplateList(ResponseBase):
    """The quantization catalog from ``compress.templates.list``.

    Not paginated: the server returns every quantizable custom model and
    its templates in one shot, alongside the supported ``base_models``.
    """

    base_models: List[str] = []
    custom_models: List[CompressCustomModel] = []

    @property
    def templates(self) -> List[CompressTemplate]:
        """Every template across all custom models, flattened for convenience."""
        return [t for m in self.custom_models for t in m.templates]


def _parse_job(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> CompressJob:
    return parse_model(
        CompressJob,
        flatten_envelope(body),
        strict=strict,
        response=response,
    )


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> CompressJobList:
    data = flatten_envelope(body)
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        # A retyped ``jobs`` would poison the ``len()`` below; empty is the
        # honest reading of "not a list", matching ``fine_tunes._parse_list``.
        jobs = []
    return parse_model(
        CompressJobList,
        {
            "page_no": data.get("page_no", 1),
            "page_size": data.get("page_size", 10),
            "total_count": data.get("total", len(jobs)),
            "jobs": jobs,
            **response_envelope(data),
        },
        strict=strict,
        response=response,
    )


def _parse_templates(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> CompressTemplateList:
    data = flatten_envelope(body)
    base_models = data.get("base_models")
    custom_models = data.get("custom_models")
    return parse_model(
        CompressTemplateList,
        {
            "base_models": base_models
            if isinstance(base_models, list)
            else [],
            "custom_models": (
                custom_models if isinstance(custom_models, list) else []
            ),
            **response_envelope(data),
        },
        strict=strict,
        response=response,
    )


def _parse_status(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> JobStatus:
    return parse_model(
        JobStatus,
        flatten_envelope(body),
        strict=strict,
        response=response,
    )


def _parse_logs(
    job_id: str,
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> JobLogs:
    data = flatten_envelope(body)
    return parse_model(
        JobLogs,
        {
            "job_id": job_id,
            "content": data.get("logs") or "",
            **response_envelope(data),
        },
        strict=strict,
        response=response,
    )


def _job_id(job: Union[CompressJob, str]) -> str:
    job_id = job.job_id if isinstance(job, CompressJob) else job
    if not job_id:
        raise DashScopeError("compress job id required")
    return job_id


def _build_create_body(
    model: str,
    template_id: str,
    *,
    job_name: Optional[str],
    job_description: Optional[str],
    hyper_parameters: Optional[Dict[str, Any]],
    custom_calibration_file_ids: Union[List[str], str, None],
    output_model_suffix: Optional[str],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": model, "template_id": template_id}
    if job_name is not None:
        if len(job_name) > 50:
            raise DashScopeError("job_name must be at most 50 characters")
        body["job_name"] = job_name
    if job_description is not None:
        if len(job_description) > 200:
            raise DashScopeError(
                "job_description must be at most 200 characters",
            )
        body["job_description"] = job_description
    if hyper_parameters is not None:
        body["hyper_parameters"] = hyper_parameters
    if custom_calibration_file_ids is not None:
        if isinstance(custom_calibration_file_ids, str):
            custom_calibration_file_ids = [custom_calibration_file_ids]
        body["custom_calibration_file_ids"] = custom_calibration_file_ids
    if output_model_suffix is not None:
        # The server names the quantized model ``{base}-{suffix}-{job_id}``
        # and rejects anything outside lowercase letters + digits, max 8.
        if not _SUFFIX_RE.fullmatch(output_model_suffix):
            raise DashScopeError(
                "output_model_suffix must be 1-8 lowercase letters/digits",
            )
        body["output_model_suffix"] = output_model_suffix
    return body


def _build_job_list_params(
    page_no: int,
    page_size: int,
    *,
    status: Optional[str],
    model: Optional[str],
    template_id: Optional[str],
    quant_spec: Optional[str],
    quant_method: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    job_name: Optional[str],
    job_id: Optional[str],
    search_key: Optional[str],
    select_key: Optional[str],
    sort_by: Optional[str],
    sort_order: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page_no": page_no, "page_size": page_size}
    for key, value in (
        ("status", status),
        ("model", model),
        ("template_id", template_id),
        ("quant_spec", quant_spec),
        ("quant_method", quant_method),
        ("start_time", start_time),
        ("end_time", end_time),
        ("job_name", job_name),
        ("job_id", job_id),
        ("search_key", search_key),
        ("select_key", select_key),
        ("sort_by", sort_by),
        ("sort_order", sort_order),
    ):
        if value is not None:
            params[key] = value
    return params


class CompressTemplates:
    """``client.fine_tunes.compress.templates`` -- list quantization templates."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    def list(
        self,
        *,
        model: Optional[str] = None,
        lang: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CompressTemplateList:
        ...

    @overload
    def list(
        self,
        *,
        model: Optional[str] = None,
        lang: Optional[str] = None,
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

    def list(
        self,
        *,
        model: Optional[str] = None,
        lang: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CompressTemplateList, "APIResponse"]:
        """List the quantization catalog (not paginated).

        Returns every quantizable custom model and its templates, plus the
        supported ``base_models``. ``model`` filters by model ID (a base
        model name returns all custom models derived from it); ``lang``
        selects ``zh-CN`` / ``en-US`` for the localized template text.
        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        params: Dict[str, Any] = {}
        if model is not None:
            params["model"] = model
        if lang is not None:
            params["lang"] = lang
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/templates",
            params=params,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        result = _parse_templates(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        return with_headers(result, resp.http_response)


class CompressJobs:
    """``client.fine_tunes.compress.jobs`` -- compression job lifecycle."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        template_id: str,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        custom_calibration_file_ids: Union[List[str], str, None] = None,
        output_model_suffix: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CompressJob:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        template_id: str,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        custom_calibration_file_ids: Union[List[str], str, None] = None,
        output_model_suffix: Optional[str] = None,
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
        template_id: str,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        custom_calibration_file_ids: Union[List[str], str, None] = None,
        output_model_suffix: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CompressJob, "APIResponse"]:
        """Create a compression (quantization) job.

        ``model`` must be a full-parameter fine-tune supported by the
        template; ``template_id`` is required. ``raw=True`` returns the
        :class:`~dashscope.transport.APIResponse` before any error check.
        """
        body = _build_create_body(
            model,
            template_id,
            job_name=job_name,
            job_description=job_description,
            hyper_parameters=hyper_parameters,
            custom_calibration_file_ids=custom_calibration_file_ids,
            output_model_suffix=output_model_suffix,
        )
        resp = self._client.transport.request_api(
            "POST",
            f"{_BASE_PATH}/jobs",
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_job(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def get(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CompressJob:
        ...

    @overload
    def get(
        self,
        job_id: str,
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

    def get(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CompressJob, "APIResponse"]:
        """Fetch one compression job.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/jobs/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_job(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        template_id: Optional[str] = None,
        quant_spec: Optional[str] = None,
        quant_method: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None,
        search_key: Optional[str] = None,
        select_key: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
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
    ) -> CompressJobList:
        ...

    @overload
    def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        template_id: Optional[str] = None,
        quant_spec: Optional[str] = None,
        quant_method: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None,
        search_key: Optional[str] = None,
        select_key: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
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

    def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        template_id: Optional[str] = None,
        quant_spec: Optional[str] = None,
        quant_method: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None,
        search_key: Optional[str] = None,
        select_key: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
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
    ) -> Union[CompressJobList, "APIResponse"]:
        """List compression jobs, one page at a time, with server-side filters.

        Filters are preserved across ``iter_pages()`` / ``next_page()``.
        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        filter_kwargs: Dict[str, Any] = {
            "status": status,
            "model": model,
            "template_id": template_id,
            "quant_spec": quant_spec,
            "quant_method": quant_method,
            "start_time": start_time,
            "end_time": end_time,
            "job_name": job_name,
            "job_id": job_id,
            "search_key": search_key,
            "select_key": select_key,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/jobs",
            params=_build_job_list_params(page_no, page_size, **filter_kwargs),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                **filter_kwargs,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            ),
            requested_page_no=page_no,
            requested_page_size=page_size,
        )
        return with_headers(result, resp.http_response)

    @overload
    def cancel(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    def cancel(
        self,
        job_id: str,
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
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Cancel a PENDING/QUEUING/RUNNING compression job.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        resp = self._client.transport.request_api(
            "POST",
            f"{_BASE_PATH}/jobs/{job_id}/cancel",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_status(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def delete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    def delete(
        self,
        job_id: str,
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

    def delete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Delete a terminal compression job; the quantized model is kept.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/jobs/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_status(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def logs(
        self,
        job_id: str,
        *,
        offset: int = 0,
        lines: int = 100,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobLogs:
        ...

    @overload
    def logs(
        self,
        job_id: str,
        *,
        offset: int = 0,
        lines: int = 100,
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

    def logs(
        self,
        job_id: str,
        *,
        offset: int = 0,
        lines: int = 100,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobLogs, "APIResponse"]:
        """Fetch logs of one compression job.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/jobs/{job_id}/logs",
            # The wire parameter is singular "line".
            params={"offset": offset, "line": lines},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_logs(
                job_id,
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    def wait(
        self,
        job: Union[CompressJob, str],
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
    ) -> CompressJob:
        """Poll until the job reaches a terminal status.

        With ``raise_on_error=False`` a transport-level failure comes back
        as the :class:`CompressJob` envelope instead of raising. ``_sleep``
        is a test seam — it replaces the pause between polls.
        """
        job_id = _job_id(job)
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, CompressJob)
            else self.get(
                job_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        )
        interval = poll_interval
        while not current.finished:
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for"
                    f" compress job {job_id}",
                )
            sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = self.get(
                job_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        return self._check_terminal(current, job_id)

    def _check_terminal(self, job: CompressJob, job_id: str) -> CompressJob:
        if job.status == FAILED and self._client.raise_on_error:
            err = job.error
            raise error_from_business_code(
                message=(
                    job.message
                    or (err.message if err else None)
                    or f"compress job {job_id} failed"
                ),
                code=job.code or (err.code if err else None),
                body=job.model_dump(exclude={"headers"}),
                headers=job.headers,
            )
        if job.status == CANCELED and self._client.raise_on_error:
            raise DashScopeError(
                f"compress job {job_id} was canceled while waiting",
                request_id=job.request_id,
            )
        return job


class Compress:
    """``client.fine_tunes.compress`` -- templates + jobs sub-resources."""

    def __init__(self, client) -> None:
        self._client = client
        self.templates = CompressTemplates(client)
        self.jobs = CompressJobs(client)


class AsyncCompressTemplates:
    """Async counterpart of :class:`CompressTemplates`."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def list(
        self,
        *,
        model: Optional[str] = None,
        lang: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CompressTemplateList:
        ...

    @overload
    async def list(
        self,
        *,
        model: Optional[str] = None,
        lang: Optional[str] = None,
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

    async def list(
        self,
        *,
        model: Optional[str] = None,
        lang: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CompressTemplateList, "APIResponse"]:
        """Async twin of :meth:`CompressTemplates.list`."""
        params: Dict[str, Any] = {}
        if model is not None:
            params["model"] = model
        if lang is not None:
            params["lang"] = lang
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/templates",
            params=params,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        result = _parse_templates(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        return with_headers(result, resp.http_response)


class AsyncCompressJobs:
    """Async counterpart of :class:`CompressJobs`."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        template_id: str,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        custom_calibration_file_ids: Union[List[str], str, None] = None,
        output_model_suffix: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CompressJob:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        template_id: str,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        custom_calibration_file_ids: Union[List[str], str, None] = None,
        output_model_suffix: Optional[str] = None,
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
        template_id: str,
        job_name: Optional[str] = None,
        job_description: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
        custom_calibration_file_ids: Union[List[str], str, None] = None,
        output_model_suffix: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CompressJob, "APIResponse"]:
        """Async twin of :meth:`CompressJobs.create`."""
        body = _build_create_body(
            model,
            template_id,
            job_name=job_name,
            job_description=job_description,
            hyper_parameters=hyper_parameters,
            custom_calibration_file_ids=custom_calibration_file_ids,
            output_model_suffix=output_model_suffix,
        )
        resp = await self._client.transport.request_api(
            "POST",
            f"{_BASE_PATH}/jobs",
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_job(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def get(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CompressJob:
        ...

    @overload
    async def get(
        self,
        job_id: str,
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

    async def get(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CompressJob, "APIResponse"]:
        """Async twin of :meth:`CompressJobs.get`."""
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/jobs/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_job(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        template_id: Optional[str] = None,
        quant_spec: Optional[str] = None,
        quant_method: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None,
        search_key: Optional[str] = None,
        select_key: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
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
    ) -> CompressJobList:
        ...

    @overload
    async def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        template_id: Optional[str] = None,
        quant_spec: Optional[str] = None,
        quant_method: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None,
        search_key: Optional[str] = None,
        select_key: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
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

    async def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        template_id: Optional[str] = None,
        quant_spec: Optional[str] = None,
        quant_method: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        job_name: Optional[str] = None,
        job_id: Optional[str] = None,
        search_key: Optional[str] = None,
        select_key: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
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
    ) -> Union[CompressJobList, "APIResponse"]:
        """Async twin of :meth:`CompressJobs.list`."""
        filter_kwargs: Dict[str, Any] = {
            "status": status,
            "model": model,
            "template_id": template_id,
            "quant_spec": quant_spec,
            "quant_method": quant_method,
            "start_time": start_time,
            "end_time": end_time,
            "job_name": job_name,
            "job_id": job_id,
            "search_key": search_key,
            "select_key": select_key,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/jobs",
            params=_build_job_list_params(page_no, page_size, **filter_kwargs),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                **filter_kwargs,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            ),
            requested_page_no=page_no,
            requested_page_size=page_size,
        )
        return with_headers(result, resp.http_response)

    @overload
    async def cancel(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    async def cancel(
        self,
        job_id: str,
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
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Async twin of :meth:`CompressJobs.cancel`."""
        resp = await self._client.transport.request_api(
            "POST",
            f"{_BASE_PATH}/jobs/{job_id}/cancel",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_status(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def delete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    async def delete(
        self,
        job_id: str,
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

    async def delete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Async twin of :meth:`CompressJobs.delete`."""
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/jobs/{job_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_status(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def logs(
        self,
        job_id: str,
        *,
        offset: int = 0,
        lines: int = 100,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobLogs:
        ...

    @overload
    async def logs(
        self,
        job_id: str,
        *,
        offset: int = 0,
        lines: int = 100,
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

    async def logs(
        self,
        job_id: str,
        *,
        offset: int = 0,
        lines: int = 100,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobLogs, "APIResponse"]:
        """Async twin of :meth:`CompressJobs.logs`."""
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/jobs/{job_id}/logs",
            # The wire parameter is singular "line".
            params={"offset": offset, "line": lines},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_logs(
                job_id,
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    async def wait(
        self,
        job: Union[CompressJob, str],
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
    ) -> CompressJob:
        """Async twin of :meth:`CompressJobs.wait`, same compat-mode contract."""
        job_id = _job_id(job)
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, CompressJob)
            else await self.get(
                job_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        )
        interval = poll_interval
        while not current.finished:
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for"
                    f" compress job {job_id}",
                )
            await sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = await self.get(
                job_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        return self._check_terminal(current, job_id)

    def _check_terminal(self, job: CompressJob, job_id: str) -> CompressJob:
        if job.status == FAILED and self._client.raise_on_error:
            err = job.error
            raise error_from_business_code(
                message=(
                    job.message
                    or (err.message if err else None)
                    or f"compress job {job_id} failed"
                ),
                code=job.code or (err.code if err else None),
                body=job.model_dump(exclude={"headers"}),
                headers=job.headers,
            )
        if job.status == CANCELED and self._client.raise_on_error:
            raise DashScopeError(
                f"compress job {job_id} was canceled while waiting",
                request_id=job.request_id,
            )
        return job


class AsyncCompress:
    """Async counterpart of :class:`Compress`."""

    def __init__(self, client) -> None:
        self._client = client
        self.templates = AsyncCompressTemplates(client)
        self.jobs = AsyncCompressJobs(client)
