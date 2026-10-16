# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.rl thin mount layer (design doc M3 3.7).

The agentic-RL business logic stays in the 1.x implementation
(``dashscope.finetune.agentic_rl``); this layer only mounts it on the
2.0 Client and routes both halves through the v2 error model -- job
management through the v2 transport, and the workflow steps by
translating 1.x exceptions into ``SdkError`` (catalog ``code`` plus
``solution``). One ``except DashScopeError`` therefore covers both.

Two halves, matching the rest of the SDK: ``rl.jobs`` is the training
job lifecycle (``create`` submits and never blocks, ``wait`` blocks,
plus ``fetch``/``list``/``cancel``/``logs``/``delete``), and ``rl``
itself carries the RL-specific workflow steps that have no
``fine_tunes`` equivalent, including the function-instance queries.

``register_functions`` / ``test_functions`` / ``upload_data`` /
``jobs.create`` / the function-instance methods are the SDK's only
public methods without ``timeout=``: the 1.x implementation they wrap
accepts none, so declaring it here would silently do nothing. The other
``jobs`` methods take ``timeout=`` as everywhere else. Those same
1.x-wrapped methods accept a per-call ``api_key`` but no ``workspace``:
the 1.x ``AgenticRL`` they delegate to has no workspace concept.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from dashscope.v2.errors import (
    DashScopeError,
    check_response,
    error_from_business_code,
    error_from_sdk_exception,
)
from dashscope.v2.resources._jobs import (
    CANCELED,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    FAILED,
)
from dashscope.v2.resources.fine_tunes import (
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
from dashscope.v2.types.common import JobStatus, with_headers

__all__ = [
    "AsyncRl",
    "AsyncRlJobs",
    "RegisteredFunctions",
    "Rl",
    "RlJobs",
    "UploadedData",
]

# Imported lazily below via the module object so tests can monkeypatch
# ``dashscope.finetune.agentic_rl.AgenticRL`` without touching 1.x code.
from dashscope.finetune import agentic_rl as _agentic_rl  # noqa: E402
from dashscope.finetune import reinforcement as _reinforcement  # noqa: E402
from dashscope.finetune.reinforcement.common import (  # noqa: E402
    errors as _rl_errors,
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


_RL_FAMILY = "sdk.agentic_rl"

# These three bases cover all 19 exception classes in common/errors.py:
# AgenticRLError roots 17, and the two *WithCode classes derive from
# RuntimeError / ValueError instead. Nothing wider is needed, since
# common/utils.py:139 already funnels transport, timeout and unexpected
# failures into RuntimeErrorWithCode; catching more would relabel bugs
# in this module as SDK errors. pydantic's ValidationError, which is
# also a ValueError, stays outside on purpose.
_RL_EXCEPTIONS = (
    _rl_errors.AgenticRLError,
    _rl_errors.RuntimeErrorWithCode,
    _rl_errors.ValueErrorWithCode,
)


@contextlib.contextmanager
def _translate_rl_errors():
    """Keep 1.x agentic-RL failures inside the v2 exception family.

    A plain sync context manager, so it wraps ``asyncio.run(...)`` and
    ``await ...`` bodies alike.
    """
    try:
        yield
    except _RL_EXCEPTIONS as exc:
        raise error_from_sdk_exception(exc, family=_RL_FAMILY) from exc


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
    """Adapt the 1.x ``FineTune`` dataclass into the v2 model.

    The 1.x submit returns ``output`` as a nested object and spells the
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

    ``create`` runs the whole 1.x workflow — register functions, upload
    datasets, submit — and returns the job without blocking; ``wait``
    polls it to a terminal status. The remaining verbs share the
    ``fine_tunes`` wire contract (RL jobs are fine-tune jobs: same
    backend, same response models) and ride the v2 client endpoint
    like every other resource; 1.x forces a management-plane host
    only in its workflow/submit path, which is a config quirk there,
    not parity.
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
        with _translate_rl_errors():
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
        return with_headers(_parse_job(response.json()), response)

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


class Rl:
    """Sync mount of the 1.x agentic-RL workflow.

    The workflow methods (``register_functions``/``test_functions``/
    ``upload_data``/``jobs.create``) wrap the 1.x async implementation
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
        with _translate_rl_errors():
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
        parameters forward to the 1.x log-pulling step that runs after
        verification.
        """
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        return with_headers(_parse_job(response.json()), response)

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


class AsyncRl:
    """Async mount of the 1.x agentic-RL workflow."""

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
        with _translate_rl_errors():
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
        parameters forward to the 1.x log-pulling step that runs after
        verification.
        """
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        with _translate_rl_errors():
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
        with _translate_rl_errors():
            return (
                await _agentic_rl.AgenticRL.query_all_function_instance_logs(
                    function_instance_id=function_instance_id,
                    page_size=page_size,
                    start_time=start_time,
                    end_time=end_time,
                    keywords=keywords,
                    api_key=api_key or self._client.config.api_key or "",
                )
            )

    async def delete_function_instance(
        self,
        function_instance_id: str,
        *,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a function-instance (raw delete API dict)."""
        with _translate_rl_errors():
            return await _agentic_rl.AgenticRL.delete_function_instance(
                function_instance_id=function_instance_id,
                api_key=api_key or self._client.config.api_key or "",
            )
