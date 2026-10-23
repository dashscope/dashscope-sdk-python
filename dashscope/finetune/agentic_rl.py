# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility shim plus the 1.x surface; the implementation moved to
``dashscope.v2.finetune.agentic_rl``.

The agentic-RL workflow is owned by the 2.0 SDK, but this import path has
to keep resolving: the ``dashscope`` CLI, the samples under
``reinforcement/examples/workspace`` and the acli skill docs all name
``dashscope.finetune.agentic_rl.AgenticRL``.

What stays here is the half of the surface that belongs to 1.x -- the
five job-management verbs, which delegate to ``dashscope.FineTunes`` (a
1.x root public API, alongside ``Deployments``), and the ``FineTune``
dataclasses they return. ``submit_job`` therefore returns a ``FineTune``
from this path and a ``SubmitResult`` from the 2.0 one; the payload and
the POST are built once, in the 2.0 class.

Import from ``dashscope.v2.finetune.agentic_rl`` in new code.
"""

from typing import Dict, List, Optional, Union

from dashscope.finetune.customize_types import (
    FineTune,
    FineTuneCancel,
    FineTuneDelete,
    FineTuneList,
)
from dashscope.finetune.finetunes import FineTunes
from dashscope.v2.finetune.agentic_rl import AgenticRL as _AgenticRLBase
from dashscope.v2.finetune.agentic_rl import SubmitOutput, SubmitResult

# Re-exported so every name this module historically carried stays
# importable from here. Both spellings resolve to the same objects: the
# reverse-reference shim under ``dashscope/finetune/reinforcement``
# aliases each submodule to the 2.0 one.
# pylint: disable-next=unused-import
from dashscope.v2.finetune.reinforcement import (  # noqa: F401
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
from dashscope.v2.finetune.reinforcement.common.errors import (  # noqa: F401
    DatasetsError,
    InstanceQueryError,
    RegistrationError,
    RuntimeErrorWithCode,
    ValidationError,
    ValueErrorWithCode,
)

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
