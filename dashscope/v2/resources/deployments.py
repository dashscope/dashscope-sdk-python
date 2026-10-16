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
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dashscope.v2.errors import check_response
from dashscope.v2.resources._sse import (
    _aiter_sse,
    _iter_sse,
    _stream_error,
    _try_json,
)
from dashscope.v2.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    response_envelope,
    with_headers,
)

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


def _parse(body: Dict[str, Any]) -> Deployment:
    return Deployment(**flatten_envelope(body))


def _parse_list(body: Dict[str, Any]) -> DeploymentList:
    data = flatten_envelope(body)
    deployments = data.get("deployments") or []
    return DeploymentList(
        page_no=data.get("page_no", 1),
        page_size=data.get("page_size", 10),
        total_count=data.get("total", len(deployments)),
        deployments=deployments,
        **response_envelope(data),
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
    ) -> Deployment:
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
        return with_headers(_parse(response.json()), response)

    def get(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Deployment:
        response = self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{deployed_model}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DeploymentList:
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

    def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Deployment:
        response = self._client.transport.request(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/scale",
            json={"deployed_model": deployed_model, "capacity": capacity},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)

    def delete(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        response = self._client.transport.request(
            "DELETE",
            f"{_BASE_PATH}/{deployed_model}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            JobStatus(**flatten_envelope(response.json())),
            response,
        )

    def stream_events(
        self,
        deployed_model_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Iterator[DeploymentEvent]:
        response = self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{deployed_model_id}/stream",
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
                    yield DeploymentEvent(
                        output=_try_json(data),
                        is_error=True,
                    )
                    return
                yield DeploymentEvent(output=_try_json(data))
        finally:
            response.close()


class AsyncDeployments:
    def __init__(self, client) -> None:
        self._client = client

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
    ) -> Deployment:
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
        return with_headers(_parse(response.json()), response)

    async def get(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Deployment:
        response = await self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{deployed_model}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)

    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DeploymentList:
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

    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: int,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Deployment:
        response = await self._client.transport.request(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/scale",
            json={"deployed_model": deployed_model, "capacity": capacity},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)

    async def delete(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        response = await self._client.transport.request(
            "DELETE",
            f"{_BASE_PATH}/{deployed_model}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            JobStatus(**flatten_envelope(response.json())),
            response,
        )

    async def stream_events(
        self,
        deployed_model_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> AsyncIterator[DeploymentEvent]:
        response = await self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{deployed_model_id}/stream",
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
                    yield DeploymentEvent(
                        output=_try_json(data),
                        is_error=True,
                    )
                    return
                yield DeploymentEvent(output=_try_json(data))
        finally:
            await response.aclose()
