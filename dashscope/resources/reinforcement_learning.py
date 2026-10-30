# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.rl thin mount layer (design doc M3 3.7).

The agentic-RL workflow lives in the 2.0 package
(``dashscope.applications.finetune``); this layer only mounts it on the 2.0 Client.
Job management and workflow steps both raise ``dashscope.errors``
natively, so one ``except DashScopeError`` covers both.

Two halves, matching the rest of the SDK: ``rl.jobs`` is the training
job lifecycle (``create`` submits and never blocks, ``wait`` blocks,
plus ``fetch``/``list``/``cancel``/``logs``/``delete``), and ``rl``
itself carries the RL-specific workflow steps that have no
``fine_tunes`` equivalent, including the function-instance queries.

The two halves also identify themselves differently on the wire, and
that split predates this layer: ``rl.jobs`` rides the v2 transport and
sends the marked ``finetune2`` module tag, while the workflow/submit
path keeps sending the unmarked ``finetune`` tag the feature has always
sent. Both are backend statistics contracts, so neither is folded into
the other.

``register_functions`` / ``test_functions`` / ``upload_data`` /
``jobs.create`` / the function-instance methods are the SDK's only
public methods without ``timeout=``: the workflow implementation they
wrap accepts none, so declaring it here would silently do nothing. The
other ``jobs`` methods take ``timeout=`` as everywhere else. Those same
workflow-wrapping methods accept a per-call ``api_key`` but no
``workspace``: ``AgenticRL`` has no workspace concept.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from dashscope.errors import (
    DashScopeError,
    check_response,
    error_from_business_code,
    response_json,
)
from dashscope.resources._jobs import (
    CANCELED,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    FAILED,
)
from dashscope.resources.fine_tunes import (
    MODULE,
    FineTune,
    FineTuneList,
    JobLogs,
    _BASE_PATH,
    _job_id,
    _parse_job,
    _parse_list,
    _parse_logs,
    _parse_status,
)
from dashscope.types.common import JobStatus, with_headers

__all__ = [
    "AsyncRl",
    "AsyncRlJobs",
    "RegisteredFunctions",
    "Rl",
    "RlJobs",
    "UploadedData",
]

# Imported below via the module object so tests can monkeypatch
# the shared application core without traversing the historical 1.x alias.
from dashscope.applications.finetune import (
    agentic_rl as _agentic_rl,
)  # noqa: E402
from dashscope.applications.finetune import (  # noqa: E402
    reinforcement as _reinforcement,
)


class RegisteredFunctions(BaseModel):
    """Ids minted by ``rl.register_functions``.

    An *entity* id names the uploaded component; an *instance* id is what
    ``rl.test_functions`` runs against.
    """

    model_config = ConfigDict(extra="allow")

    rollout_entity_ids: List[str] = []
    reward_entity_ids: List[str] = []
    group_reward_entity_ids: List[str] = []
    rollout_instance_ids: List[str] = []
    reward_instance_ids: List[str] = []
    group_reward_instance_ids: List[str] = []


class UploadedData(BaseModel):
    """Ids minted by ``rl.upload_data``.

    Named after the parameters of ``fine_tunes.create`` so the result can be
    forwarded straight through.
    """

    model_config = ConfigDict(extra="allow")

    training_file_ids: List[str] = []
    validation_file_ids: List[str] = []


def _resolve_func_type(func_type: Union[str, Any]) -> Any:
    if isinstance(func_type, str):
        return _reinforcement.FunctionType[func_type.upper()]
    return func_type


def _new_agentic_rl(
    client,
    config_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    rl = _agentic_rl.AgenticRL(
        api_key=api_key or client.config.api_key or "",
    )
    if config_path:
        rl.init(config_path=config_path)
    return rl


def _to_job(result: Any) -> FineTune:
    """Adapt the RL submit result into the v2 job model.

    The submit returns ``output`` as a nested object and spells the
    timestamps ``create_time`` / ``end_time``; the v2 model's aliases
    absorb both once the envelope is flattened by hand here.
    """
    body: Dict[str, Any] = {
        "request_id": getattr(result, "request_id", None),
        "status_code": getattr(result, "status_code", None),
        "code": getattr(result, "code", None),
        "message": getattr(result, "message", None),
    }
    output = getattr(result, "output", None)
    if output is not None:
        body.update(vars(output))
    return FineTune(**body)


def _check_terminal(client, job: FineTune, job_id: str) -> FineTune:
    """Mirror ``FineTunes._check_terminal`` for the RL polling loop."""
    if job.status == FAILED and client.raise_on_error:
        raise error_from_business_code(
            message=job.message or f"fine-tune job {job_id} failed",
            code=job.code,
            # ``headers`` is transport metadata, not wire body.
            body=job.model_dump(exclude={"headers"}),
            headers=job.headers,
        )
    if job.status == CANCELED and client.raise_on_error:
        raise DashScopeError(
            f"fine-tune job {job_id} was canceled while waiting",
            request_id=job.request_id,
        )
    return job


async def _register_functions_workflow(
    rl,
    rollout_classpaths: Optional[List[str]],
    reward_classpaths: Optional[List[str]],
    group_reward_classpaths: Optional[List[str]],
    workspace_dir: str,
    lazy_load: bool,
) -> RegisteredFunctions:
    if rollout_classpaths or reward_classpaths or group_reward_classpaths:
        rl.tuning.functions = []
        rl.tuning.add_function_components(
            functype=_reinforcement.FunctionType.ROLLOUT,
            classpaths=rollout_classpaths,
            workspace_dir=workspace_dir,
        )
        rl.tuning.add_function_components(
            functype=_reinforcement.FunctionType.REWARD,
            classpaths=reward_classpaths,
            workspace_dir=workspace_dir,
        )
        rl.tuning.add_function_components(
            functype=_reinforcement.FunctionType.GROUP_REWARD,
            classpaths=group_reward_classpaths,
            workspace_dir=workspace_dir,
        )
    (
        rollout_entity_ids,
        reward_entity_ids,
        group_reward_entity_ids,
        rollout_instance_ids,
        reward_instance_ids,
        group_reward_instance_ids,
    ) = await rl.register_functions(lazy_load=lazy_load)
    return RegisteredFunctions(
        rollout_entity_ids=rollout_entity_ids,
        reward_entity_ids=reward_entity_ids,
        group_reward_entity_ids=group_reward_entity_ids,
        rollout_instance_ids=rollout_instance_ids,
        reward_instance_ids=reward_instance_ids,
        group_reward_instance_ids=group_reward_instance_ids,
    )


class RlJobs:
    """``client.rl.jobs``: the agentic-RL training job lifecycle.

    ``create`` runs the whole workflow — register functions, upload
    datasets, submit — and returns the job without blocking; ``wait``
    polls it to a terminal status. The remaining verbs share the
    ``fine_tunes`` wire contract (RL jobs are fine-tune jobs: same
    backend, same response models) and ride the v2 client endpoint
    like every other resource. The workflow/submit path is the
    exception: it posts to the management-plane host its own constants
    name, as it always has.
    """

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        config_path: Optional[str] = None,
        job_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> FineTune:
        """Submit the workflow and return immediately; ``wait`` blocks.

        ``kwargs`` override individual keys of the training config loaded
        from ``config_path``, which is open-ended by design.
        """
        rl = _new_agentic_rl(self._client, api_key=api_key)
        if job_name is not None:
            kwargs["job_name"] = job_name
        rl.init(config_path=config_path, **kwargs)
        result = asyncio.run(rl.run())
        return _to_job(result)

    def fetch(
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
        return with_headers(_parse_job(response_json(response)), response)

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

        Polls through :meth:`fetch` on the client endpoint. With
        ``raise_on_error=False`` a transport-level failure comes back
        as the :class:`FineTune` envelope instead of raising.
        ``_sleep`` is a test seam — it replaces the pause between polls.
        """
        job_id = _job_id(job)
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, FineTune)
            else self.fetch(job_id, api_key=api_key, workspace=workspace)
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
            current = self.fetch(job_id, api_key=api_key, workspace=workspace)
        return _check_terminal(self._client, current, job_id)

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
        return with_headers(_parse_list(response_json(response)), response)

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
        return with_headers(_parse_status(response_json(response)), response)

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
        return with_headers(_parse_status(response_json(response)), response)

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
        return with_headers(
            _parse_logs(job_id, response_json(response)),
            response,
        )


class Rl:
    """Sync mount of the agentic-RL workflow.

    The workflow methods (``register_functions``/``test_functions``/
    ``upload_data``/``jobs.create``) wrap the async implementation
    with ``asyncio.run``, so they must not be called from a thread that
    already runs an event loop (Jupyter, FastAPI, pytest-asyncio) —
    use :class:`AsyncRl` there instead.
    """

    def __init__(self, client) -> None:
        self._client = client
        self.jobs = RlJobs(client)

    def register_functions(
        self,
        *,
        rollout_classpaths: Optional[List[str]] = None,
        reward_classpaths: Optional[List[str]] = None,
        group_reward_classpaths: Optional[List[str]] = None,
        workspace_dir: str = "./",
        lazy_load: bool = True,
        api_key: Optional[str] = None,
    ) -> RegisteredFunctions:
        rl = _new_agentic_rl(self._client, api_key=api_key)
        return asyncio.run(
            _register_functions_workflow(
                rl,
                rollout_classpaths,
                reward_classpaths,
                group_reward_classpaths,
                workspace_dir,
                lazy_load,
            ),
        )

    def test_functions(
        self,
        *,
        instance_id: str,
        func_type: Union[str, Any],
        input_data: Dict[str, Any],
        pull_logs: bool = False,
        log_page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """Returns whatever the customer's own function returns.

        The ``pull_logs``/``log_page_size``/``start_time``/``end_time``
        parameters forward to the log-pulling step that runs after
        verification.
        """
        return asyncio.run(
            _agentic_rl.AgenticRL.test_functions(
                instance_id=instance_id,
                functype=_resolve_func_type(func_type),
                input_data=input_data,
                api_key=api_key or self._client.config.api_key or "",
                pull_logs=pull_logs,
                log_page_size=log_page_size,
                start_time=start_time,
                end_time=end_time,
            ),
        )

    def upload_data(
        self,
        training_files: Union[List[str], str],
        validation_files: Union[List[str], str, None] = None,
        *,
        api_key: Optional[str] = None,
    ) -> UploadedData:
        rl = _new_agentic_rl(self._client, api_key=api_key)
        train_ids, val_ids = asyncio.run(
            rl.upload_datasets(
                training_files=training_files,
                validation_files=validation_files,
            ),
        )
        return UploadedData(
            training_file_ids=train_ids,
            validation_file_ids=val_ids or [],
        )

    def query_function_instance_logs(
        self,
        function_instance_id: str,
        *,
        page_number: int = 1,
        page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One page of function-instance logs (raw log-query API dict).

        Wraps ``asyncio.run``, so it must not be called from a running
        event loop — use :class:`AsyncRl` there.
        """
        return asyncio.run(
            _agentic_rl.AgenticRL.query_function_instance_logs(
                function_instance_id=function_instance_id,
                page_number=page_number,
                page_size=page_size,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
                api_key=api_key or self._client.config.api_key or "",
            ),
        )

    def query_all_function_instance_logs(
        self,
        function_instance_id: str,
        *,
        page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ) -> List[str]:
        """All function-instance logs, aggregated across pages.

        Wraps ``asyncio.run``, so it must not be called from a running
        event loop — use :class:`AsyncRl` there.
        """
        return asyncio.run(
            _agentic_rl.AgenticRL.query_all_function_instance_logs(
                function_instance_id=function_instance_id,
                page_size=page_size,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
                api_key=api_key or self._client.config.api_key or "",
            ),
        )

    def delete_function_instance(
        self,
        function_instance_id: str,
        *,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a function-instance (raw delete API dict).

        Wraps ``asyncio.run``, so it must not be called from a running
        event loop — use :class:`AsyncRl` there.
        """
        return asyncio.run(
            _agentic_rl.AgenticRL.delete_function_instance(
                function_instance_id=function_instance_id,
                api_key=api_key or self._client.config.api_key or "",
            ),
        )


class AsyncRlJobs:
    """Async ``client.rl.jobs``."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        config_path: Optional[str] = None,
        job_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> FineTune:
        """Submit the workflow and return immediately; ``wait`` blocks.

        ``kwargs`` override individual keys of the training config loaded
        from ``config_path``, which is open-ended by design.
        """
        rl = _new_agentic_rl(self._client, api_key=api_key)
        if job_name is not None:
            kwargs["job_name"] = job_name
        rl.init(config_path=config_path, **kwargs)
        result = await rl.run()
        return _to_job(result)

    async def fetch(
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
        return with_headers(_parse_job(response_json(response)), response)

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
        """Poll until the job reaches a terminal status.

        Polls through :meth:`fetch` on the client endpoint. With
        ``raise_on_error=False`` a transport-level failure comes back
        as the :class:`FineTune` envelope instead of raising.
        ``_sleep`` is a test seam — it replaces the pause between polls.
        """
        job_id = _job_id(job)
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, FineTune)
            else await self.fetch(job_id, api_key=api_key, workspace=workspace)
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
            current = await self.fetch(
                job_id,
                api_key=api_key,
                workspace=workspace,
            )
        return _check_terminal(self._client, current, job_id)

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
        return with_headers(_parse_list(response_json(response)), response)

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
        return with_headers(_parse_status(response_json(response)), response)

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
        return with_headers(_parse_status(response_json(response)), response)

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
        return with_headers(
            _parse_logs(job_id, response_json(response)),
            response,
        )


class AsyncRl:
    """Async mount of the agentic-RL workflow."""

    def __init__(self, client) -> None:
        self._client = client
        self.jobs = AsyncRlJobs(client)

    async def register_functions(
        self,
        *,
        rollout_classpaths: Optional[List[str]] = None,
        reward_classpaths: Optional[List[str]] = None,
        group_reward_classpaths: Optional[List[str]] = None,
        workspace_dir: str = "./",
        lazy_load: bool = True,
        api_key: Optional[str] = None,
    ) -> RegisteredFunctions:
        rl = _new_agentic_rl(self._client, api_key=api_key)
        return await _register_functions_workflow(
            rl,
            rollout_classpaths,
            reward_classpaths,
            group_reward_classpaths,
            workspace_dir,
            lazy_load,
        )

    async def test_functions(
        self,
        *,
        instance_id: str,
        func_type: Union[str, Any],
        input_data: Dict[str, Any],
        pull_logs: bool = False,
        log_page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """Returns whatever the customer's own function returns.

        The ``pull_logs``/``log_page_size``/``start_time``/``end_time``
        parameters forward to the log-pulling step that runs after
        verification.
        """
        return await _agentic_rl.AgenticRL.test_functions(
            instance_id=instance_id,
            functype=_resolve_func_type(func_type),
            input_data=input_data,
            api_key=api_key or self._client.config.api_key or "",
            pull_logs=pull_logs,
            log_page_size=log_page_size,
            start_time=start_time,
            end_time=end_time,
        )

    async def upload_data(
        self,
        training_files: Union[List[str], str],
        validation_files: Union[List[str], str, None] = None,
        *,
        api_key: Optional[str] = None,
    ) -> UploadedData:
        rl = _new_agentic_rl(self._client, api_key=api_key)
        train_ids, val_ids = await rl.upload_datasets(
            training_files=training_files,
            validation_files=validation_files,
        )
        return UploadedData(
            training_file_ids=train_ids,
            validation_file_ids=val_ids or [],
        )

    async def query_function_instance_logs(
        self,
        function_instance_id: str,
        *,
        page_number: int = 1,
        page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One page of function-instance logs (raw log-query API dict)."""
        return await _agentic_rl.AgenticRL.query_function_instance_logs(
            function_instance_id=function_instance_id,
            page_number=page_number,
            page_size=page_size,
            start_time=start_time,
            end_time=end_time,
            keywords=keywords,
            api_key=api_key or self._client.config.api_key or "",
        )

    async def query_all_function_instance_logs(
        self,
        function_instance_id: str,
        *,
        page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ) -> List[str]:
        """All function-instance logs, aggregated across pages."""
        return await _agentic_rl.AgenticRL.query_all_function_instance_logs(
            function_instance_id=function_instance_id,
            page_size=page_size,
            start_time=start_time,
            end_time=end_time,
            keywords=keywords,
            api_key=api_key or self._client.config.api_key or "",
        )

    async def delete_function_instance(
        self,
        function_instance_id: str,
        *,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a function-instance (raw delete API dict)."""
        return await _agentic_rl.AgenticRL.delete_function_instance(
            function_instance_id=function_instance_id,
            api_key=api_key or self._client.config.api_key or "",
        )
