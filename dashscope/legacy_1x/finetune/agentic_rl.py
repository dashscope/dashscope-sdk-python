# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# pylint: disable=unused-import
"""Compatibility shim plus the 1.x surface; the implementation moved to
``dashscope.applications.finetune.agentic_rl``.

The agentic-RL workflow is owned by the 2.0 SDK, while the ``dashscope``
CLI and acli skill docs use this 1.x import path.

What stays here is the half of the surface that belongs to 1.x -- the
five job-management verbs, which delegate to ``dashscope.FineTunes`` (a
1.x root public API, alongside ``Deployments``), and the ``FineTune``
dataclasses they return. ``submit_job`` therefore returns a ``FineTune``
from this path and a ``SubmitResult`` from the 2.0 one; the payload and
the POST are built once, in the 2.0 class.

New 2.0 code reaches the workflow through ``client.rl``.
"""

import functools
import inspect
from typing import Dict, List, Optional, Union, cast

from dashscope.legacy_1x.finetune.customize_types import (
    FineTune,
    FineTuneCancel,
    FineTuneDelete,
    FineTuneList,
)
from dashscope.legacy_1x.finetune.finetunes import FineTunes
from dashscope.applications.finetune.agentic_rl import (
    AgenticRL as _AgenticRLBase,
)
from dashscope.applications.finetune.agentic_rl import (
    SubmitOutput,
    SubmitResult,
)

# Re-export every name this module historically carried.
# pylint: disable-next=unused-import
from dashscope.applications.finetune.reinforcement import (  # noqa: F401
    DASHSCOPE_HTTP_BASE_URL,
    AgenticRLFunctionComponent,
    AgenticRLTuning,
    Dataset,
    DatasetsType,
    FunctionType,
    GroupRewardInput,
    RewardFunctionComponent,
    RewardInput,
    RolloutFunctionComponent,
    RolloutInput,
    TrainingDataset,
    TrainingType,
    TuningModel,
    ValidationDataset,
    deep_remove_none,
    generate_random_id,
    get_func_type_id,
    logger,
    set_api_key,
)

# pylint: disable-next=unused-import
from dashscope.applications.finetune.reinforcement.common.errors import (
    AgenticRLError as _CoreAgenticRLError,
)
from dashscope.legacy_1x.finetune.reinforcement.common import (
    errors as _legacy_errors,
)
from dashscope.legacy_1x.finetune.reinforcement.common.errors import (  # noqa: F401
    AgenticRLError,
    DatasetsError,
    InstanceQueryError,
    RegistrationError,
    RuntimeErrorWithCode,
    ValidationError,
    ValueErrorWithCode,
)


def _to_legacy_error(exc: _CoreAgenticRLError) -> AgenticRLError:
    """Rehydrate a core v2 error as the historical 1.x class.

    Bypassing ``__init__`` avoids applying the old message prefixes twice;
    the shared core already populated the complete compatibility attributes.
    """
    legacy_cls = getattr(
        _legacy_errors,
        type(exc).__name__,
        AgenticRLError,
    )
    if not isinstance(legacy_cls, type) or not issubclass(
        legacy_cls,
        BaseException,
    ):
        legacy_cls = AgenticRLError
    legacy = legacy_cls.__new__(legacy_cls)
    Exception.__init__(legacy, *exc.args)
    legacy.__dict__.update(exc.__dict__)
    legacy.message = getattr(exc, "message", str(exc))
    legacy.error_code = getattr(exc, "error_code", 1000)
    return cast(AgenticRLError, legacy)


def _translate_sync(method):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except _CoreAgenticRLError as exc:
            raise _to_legacy_error(exc) from exc

    return wrapped


def _translate_async(method):
    @functools.wraps(method)
    async def wrapped(*args, **kwargs):
        try:
            return await method(*args, **kwargs)
        except _CoreAgenticRLError as exc:
            raise _to_legacy_error(exc) from exc

    return wrapped


__all__ = [
    "AgenticRL",
    "DASHSCOPE_HTTP_BASE_URL",
    "FineTune",
    "FineTuneCancel",
    "FineTuneDelete",
    "FineTuneList",
    "FineTunes",
    "SubmitOutput",
    "SubmitResult",
]


class AgenticRL(_AgenticRLBase):
    """The 1.x surface: ``FineTune`` returns plus the ``FineTunes`` verbs.

    The five verbs below are 1.x-only. ``Client(...).rl.jobs`` already
    implements the same lifecycle natively on the 2.0 transport, so
    duplicating them into the 2.0 class would have given the SDK two
    copies of every job verb.
    """

    def __init__(self, api_key: str = None):
        # Keep the historical patch point and old error type. The core class
        # uses the same AgenticRLTuning implementation, so no business logic
        # is duplicated here.
        super(_AgenticRLBase, self).__init__()
        try:
            set_api_key(api_key)
        except Exception as exc:
            raise ValueErrorWithCode(
                "Invalid API key configuration",
                error_code=3001,
            ) from exc
        self._api_key = api_key

    def submit_job(  # type: ignore[override]
        self,
        model: Optional[str] = None,
        datasets: Optional[List[Dataset]] = None,
        functions: Optional[
            Union[
                List[
                    Union[
                        RolloutFunctionComponent,
                        RewardFunctionComponent,
                        AgenticRLFunctionComponent,
                    ]
                ],
                RolloutFunctionComponent,
                RewardFunctionComponent,
                AgenticRLFunctionComponent,
            ]
        ] = None,
        hyper_parameters: Optional[Dict[str, str]] = None,
        resources: Optional[Dict[str, str]] = None,
        job_name: Optional[str] = None,
        teacher_model: Optional[str] = None,
        **kwargs,
    ) -> FineTune:
        """
        Submit RL tuning job to the platform.
        """
        return FineTune(
            **self._submit_envelope(
                model=model,
                datasets=datasets,
                functions=functions,
                hyper_parameters=hyper_parameters,
                resources=resources,
                job_name=job_name,
                teacher_model=teacher_model,
                **kwargs,
            ),
        )

    @classmethod
    def cancel(
        cls,
        job_id: str,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> FineTuneCancel:
        """Cancel a running fine-tune job."""
        kwargs["base_address"] = DASHSCOPE_HTTP_BASE_URL

        return FineTunes.cancel(
            job_id,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )

    @classmethod
    def list(
        cls,
        page_no=1,
        page_size=10,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> FineTuneList:
        """List fine-tune jobs."""
        kwargs["base_address"] = DASHSCOPE_HTTP_BASE_URL

        return FineTunes.list(
            page_no=page_no,
            page_size=page_size,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )

    @classmethod
    def get(
        cls,
        job_id: str,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> FineTune:
        """Get fine-tune job information."""
        kwargs["base_address"] = DASHSCOPE_HTTP_BASE_URL

        return FineTunes.get(
            job_id,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )

    @classmethod
    def delete(
        cls,
        job_id: str,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> FineTuneDelete:
        """Delete a fine-tune job."""
        kwargs["base_address"] = DASHSCOPE_HTTP_BASE_URL

        return FineTunes.delete(
            job_id,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )

    @classmethod
    def logs(
        cls,
        job_id: str,
        offset: int = 1,
        lines: int = 1000,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> FineTune:
        """Get job logs."""
        kwargs["base_address"] = DASHSCOPE_HTTP_BASE_URL

        return FineTunes.logs(
            job_id,
            offset=offset,
            line=lines,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )


# Every workflow entry point crosses the facade boundary here. The shared core
# raises the 2.0 family; the historical class translates it back to the exact
# 1.x exception class before it reaches user code. functools.wraps preserves
# the published signatures for inspect/help tooling.
for _method_name in ("init", "register_functions", "upload_datasets", "run"):
    _method = getattr(_AgenticRLBase, _method_name)
    _wrapper = (
        _translate_async(_method)
        if inspect.iscoroutinefunction(_method)
        else _translate_sync(_method)
    )
    setattr(AgenticRL, _method_name, _wrapper)

AgenticRL.submit_job = _translate_sync(AgenticRL.submit_job)

for _method_name in (
    "test_functions",
    "query_function_instance_logs",
    "query_all_function_instance_logs",
    "delete_function_instance",
):
    _descriptor = _AgenticRLBase.__dict__[_method_name]
    _method = _descriptor.__func__
    setattr(AgenticRL, _method_name, classmethod(_translate_async(_method)))

del _descriptor, _method, _method_name, _wrapper
