# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.deployments resource (design doc M3 3.6).

Wire contract follows the 1.x ``Deployments`` REST surface:
``POST /deployments``, ``GET /deployments``, ``GET /deployments/{id}``,
``PUT /deployments/{id}/scale``, ``DELETE /deployments/{id}``,
``GET /deployments/{id}/stream`` (SSE deployment events).
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dashscope.resources._sse import (
    _aiter_sse,
    _iter_sse,
    _log_stream_error,
    _stream_error,
    _try_json,
)
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    response_envelope,
    with_headers,
)

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_BASE_PATH = "/deployments"

# 1.x tags this API "finetune" (Deployments lives in dashscope.finetune).
MODULE = "finetune"

__all__ = [
    "AsyncDeployments",
    "Deployment",
    "DeploymentEvent",
    "DeploymentList",
    "Deployments",
]


class Deployment(ResponseBase):
    """One model deployment.

    The same type comes back from ``create`` / ``get`` / ``scale`` and is
    nested in ``DeploymentList.deployments``, so there is one shape to learn;
    the response envelope therefore rides along on the nested copies too,
    empty there because the server reports it once per response.
    """

    deployed_model: Optional[str] = None
    model: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("model", "model_name"),
    )
    base_model: Optional[str] = None
    status: Optional[str] = None
    capacity: Optional[int] = None
    base_capacity: Optional[int] = None
    ready_capacity: Optional[int] = None
    workspace_id: Optional[str] = None
    charge_type: Optional[str] = None
    created_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("created_at", "gmt_create"),
    )
    updated_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("updated_at", "gmt_modified"),
    )
    modifier: Optional[str] = None
    creator: Optional[str] = None


class DeploymentList(ResponseBase):
    page_no: int = 1
    page_size: int = 10
    total_count: int = 0
    deployments: List[Deployment] = []


class DeploymentEvent(BaseModel):
    """One SSE frame from ``/deployments/{id}/stream``."""

    model_config = ConfigDict(extra="allow")

    output: Union[Dict[str, Any], str, None] = None
    is_error: bool = False


def _parse(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> Deployment:
    return parse_model(
        Deployment,
        flatten_envelope(body),
        strict=strict,
        response=response,
    )


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> DeploymentList:
    data = flatten_envelope(body)
    deployments = data.get("deployments")
    if not isinstance(deployments, list):
        # A retyped ``deployments`` would poison the ``len()`` below (a
        # string's length is not a page total). Empty is the honest
        # reading of "not a list", matching ``models._parse_list``.
        deployments = []
    return parse_model(
        DeploymentList,
        {
            "page_no": data.get("page_no", 1),
            "page_size": data.get("page_size", 10),
            "total_count": data.get("total", len(deployments)),
            "deployments": deployments,
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


def _parse_event(
    data: Any,
    *,
    is_error: bool = False,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> DeploymentEvent:
    """One SSE frame, degrading rather than breaking the iteration.

    A drifted frame used to raise a bare ``ValidationError`` partway
    through the generator -- earlier frames were already consumed and
    ``except DashScopeError`` could not catch it. ``response`` is the
    stream itself, which is the only response a frame belongs to.
    """
    return parse_model(
        DeploymentEvent,
        {"output": _try_json(data), "is_error": is_error},
        strict=strict,
        response=response,
    )


def _build_create_body(
    model: str,
    capacity: int,
    model_version: Optional[str],
    suffix: Optional[str],
    plan: Optional[str],
    template_id: Optional[str],
    deploy_spec: Optional[Dict[str, Any]],
    billing_method: Optional[str],
) -> Dict[str, Any]:
    # Wire spellings; the public names are ``model`` / ``model_version``.
    body: Dict[str, Any] = {"model_name": model, "capacity": capacity}
    if model_version is not None:
        body["model_version"] = model_version
    if suffix is not None:
        body["suffix"] = suffix
    if plan is not None:
        body["plan"] = plan
    if template_id is not None:
        body["template_id"] = template_id

    # MU deployments reject the create call unless both are present.
    if deploy_spec is not None:
        body["deploy_spec"] = deploy_spec
    if billing_method is not None:
        body["billing_method"] = billing_method
    return body


class Deployments:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        capacity: int,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        capacity: int,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
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
        capacity: int,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Create a deployment of ``model`` with ``capacity`` instances.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_create_body(
            model,
            capacity,
            model_version,
            suffix,
            plan,
            template_id,
            deploy_spec,
            billing_method,
        )
        resp = self._client.transport.request_api(
            "POST",
            _BASE_PATH,
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
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def get(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def get(
        self,
        deployed_model: str,
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
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Fetch one deployment by ``deployed_model`` id.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{deployed_model}",
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
            _parse(
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
    ) -> DeploymentList:
        ...

    @overload
    def list(
        self,
        *,
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
    ) -> Union[DeploymentList, "APIResponse"]:
        """List deployments, one page at a time.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
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
            _parse_list(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
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

    def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Resize a deployment to ``capacity`` instances.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/scale",
            json={"deployed_model": deployed_model, "capacity": capacity},
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
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def delete(
        self,
        deployed_model: str,
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
        deployed_model: str,
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
        deployed_model: str,
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
        """Delete a deployment; the job status reports the teardown.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{deployed_model}",
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

    def stream_events(
        self,
        deployed_model_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Iterator[DeploymentEvent]:
        response = self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{deployed_model_id}/stream",
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
        )
        try:
            for is_error, status_code, data in _iter_sse(response):
                if is_error:
                    if self._client.raise_on_error:
                        raise _stream_error(
                            status_code,
                            data,
                            response.headers,
                        )
                    _log_stream_error(data, response.headers)
                    yield _parse_event(
                        data,
                        is_error=True,
                        strict=self._client.config.strict_validation,
                        response=response,
                    )
                    return
                yield _parse_event(
                    data,
                    strict=self._client.config.strict_validation,
                    response=response,
                )
        finally:
            response.close()


class AsyncDeployments:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        capacity: int,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        capacity: int,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
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
        capacity: int,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Create a deployment of ``model`` with ``capacity`` instances.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_create_body(
            model,
            capacity,
            model_version,
            suffix,
            plan,
            template_id,
            deploy_spec,
            billing_method,
        )
        resp = await self._client.transport.request_api(
            "POST",
            _BASE_PATH,
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
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def get(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def get(
        self,
        deployed_model: str,
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
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Fetch one deployment by ``deployed_model`` id.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{deployed_model}",
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
            _parse(
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
    ) -> DeploymentList:
        ...

    @overload
    async def list(
        self,
        *,
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
    ) -> Union[DeploymentList, "APIResponse"]:
        """List deployments, one page at a time.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
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
            _parse_list(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
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

    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Resize a deployment to ``capacity`` instances.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/scale",
            json={"deployed_model": deployed_model, "capacity": capacity},
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
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def delete(
        self,
        deployed_model: str,
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
        deployed_model: str,
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
        deployed_model: str,
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
        """Delete a deployment; the job status reports the teardown.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{deployed_model}",
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

    async def stream_events(
        self,
        deployed_model_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> AsyncIterator[DeploymentEvent]:
        response = await self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{deployed_model_id}/stream",
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
        )
        try:
            async for is_error, status_code, data in _aiter_sse(response):
                if is_error:
                    if self._client.raise_on_error:
                        raise _stream_error(
                            status_code,
                            data,
                            response.headers,
                        )
                    _log_stream_error(data, response.headers)
                    yield _parse_event(
                        data,
                        is_error=True,
                        strict=self._client.config.strict_validation,
                        response=response,
                    )
                    return
                yield _parse_event(
                    data,
                    strict=self._client.config.strict_validation,
                    response=response,
                )
        finally:
            await response.aclose()
