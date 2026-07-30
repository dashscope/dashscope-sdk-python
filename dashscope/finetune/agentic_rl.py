# -*- coding: utf-8 -*-
from __future__ import annotations

# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Union, List, Optional, ClassVar, Dict, Any, Tuple
from typing_extensions import Self

from dashscope.client.base_api import CreateMixin
from dashscope.finetune.customize_types import (
    FineTune,
    FineTuneCancel,
    FineTuneDelete,
    FineTuneList,
)
from dashscope.finetune.finetunes import FineTunes
from dashscope.finetune.reinforcement import (
    AgenticRLFunctionComponent,
    RolloutFunctionComponent,
    RewardFunctionComponent,
    Dataset,
    TrainingDataset,
    ValidationDataset,
)
from dashscope.finetune.reinforcement import AgenticRLTuning, TuningModel
from dashscope.finetune.reinforcement import DASHSCOPE_HTTP_BASE_URL
from dashscope.finetune.reinforcement import (
    FunctionType,
    DatasetsType,
)
from dashscope.finetune.reinforcement import (
    RewardInput,
    RolloutInput,
    GroupRewardInput,
)
from dashscope.finetune.reinforcement import logger
from dashscope.finetune.reinforcement import (
    set_api_key,
    generate_random_id,
    get_func_type_id,
    deep_remove_none,
)
from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    DashScopeException,
)
from dashscope.common.error_registry import (
    INVALID_API_KEY,
    INVALID_REQUEST,
    INTERNAL_ERROR,
    REQUEST_TIMEOUT,
    AGENTIC_RL_INVALID_API_KEY,
    AGENTIC_RL_FUNCTION_REGISTRATION_FAILED,
    AGENTIC_RL_DATASETS_UPLOAD_FAILED,
    AGENTIC_RL_DUPLICATE_FUNCTION_NAMES,
    AGENTIC_RL_JOB_SUBMISSION_FAILED,
    AGENTIC_RL_WORKFLOW_FAILED,
    AGENTIC_RL_UNSUPPORTED_FUNCTION_TYPE,
    AGENTIC_RL_FUNCTION_TEST_FAILED,
    AGENTIC_RL_FUNCTION_TEST_TIMEOUT,
)
from dashscope.finetune.reinforcement.common.errors import (
    BasePermissionError,
    ConfigurationError,
    InputError,
    ValidationError,
)

# 4xx exception types — client-side errors
_CLIENT_ERROR_TYPES = (
    BasePermissionError,
    ConfigurationError,
    ValidationError,
    InputError,
)


def _is_client_error(exc: Exception) -> bool:
    """Check if an exception is a client-side error (should map to 4xx).

    Per dashscope public-errors spec:
    - BasePermissionError / ConfigurationError / ValidationError / InputError
      -> 4xx (client-side)
    - All other exceptions -> 5xx (server-side)
    """
    return isinstance(exc, _CLIENT_ERROR_TYPES)


class AgenticRL(AgenticRLTuning, CreateMixin):
    SUB_PATH: ClassVar[str] = "fine-tunes"

    def __init__(self, api_key: str = None):
        super().__init__()

        try:
            set_api_key(api_key)
        except Exception as e:
            logger.error(
                "[%s] %s | %s",
                AGENTIC_RL_INVALID_API_KEY.name,
                AGENTIC_RL_INVALID_API_KEY.format_message(),
                e,
                exc_info=True,
            )
            exc = AuthenticationError(INVALID_API_KEY.format_msg())
            exc.status_code = INVALID_API_KEY.status_code
            exc.error_code = INVALID_API_KEY.error_code
            raise exc from e

    def init(self, config_path: Optional[str] = None, **kwargs) -> Self:
        """
        Initialize an AgenticRL instance from a YAML configuration file.
        """
        self.tuning = TuningModel.load_from_yaml(config_path or "", **kwargs)

        return self

    async def register_functions(
        self,
        functions: Optional[
            Union[
                List[Union[RolloutFunctionComponent, RewardFunctionComponent]],
                RolloutFunctionComponent,
                RewardFunctionComponent,
            ]
        ] = None,
        lazy_load: Optional[bool] = True,
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
    ]:
        """Register function components and return entity/instance IDs."""
        if functions:
            self.tuning.functions = functions

        try:
            (
                rollout_entity_ids,
                reward_entity_ids,
                group_reward_entity_ids,
                rollout_instance_ids,
                reward_instance_ids,
                group_reward_instance_ids,
            ) = await self.tuning.register_functions(
                lazy_load=lazy_load,
            )
            logger.info("Function components registered")
        except Exception as e:
            if isinstance(e, DashScopeException):
                raise
            logger.error(
                "[%s] %s | %s",
                AGENTIC_RL_FUNCTION_REGISTRATION_FAILED.name,
                AGENTIC_RL_FUNCTION_REGISTRATION_FAILED.format_message(),
                e,
                exc_info=True,
            )
            original_summary = f"{type(e).__name__}: {str(e)[:150]}"
            if _is_client_error(e):
                public_error = INVALID_REQUEST
                exc = InvalidParameter(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            else:
                public_error = INTERNAL_ERROR
                exc = DashScopeException(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            exc.status_code = public_error.status_code
            exc.error_code = public_error.error_code
            raise exc from e

        return (
            rollout_entity_ids,
            reward_entity_ids,
            group_reward_entity_ids,
            rollout_instance_ids,
            reward_instance_ids,
            group_reward_instance_ids,
        )

    async def upload_datasets(
        self,
        datasets: Optional[List[Dataset]] = None,
        training_files: Optional[Union[List[str], str]] = None,
        validation_files: Optional[Union[List[str], str]] = None,
    ) -> Tuple[List[str], List[str]]:
        if datasets:
            self.tuning.datasets = datasets

        try:
            (
                uploaded_training_ids,
                uploaded_validation_ids,
            ) = await self.tuning.upload_datasets(
                training_files=training_files or [],
                validation_files=validation_files or [],
            )
            logger.info("Datasets uploaded")
        except Exception as e:
            if isinstance(e, DashScopeException):
                raise
            logger.error(
                "[%s] %s | %s",
                AGENTIC_RL_DATASETS_UPLOAD_FAILED.name,
                AGENTIC_RL_DATASETS_UPLOAD_FAILED.format_message(),
                e,
                exc_info=True,
            )
            original_summary = f"{type(e).__name__}: {str(e)[:150]}"
            if _is_client_error(e):
                public_error = INVALID_REQUEST
                exc = InvalidParameter(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            else:
                public_error = INTERNAL_ERROR
                exc = DashScopeException(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            exc.status_code = public_error.status_code
            exc.error_code = public_error.error_code
            raise exc from e

        return uploaded_training_ids, uploaded_validation_ids

    def submit_job(
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
        **kwargs,
    ) -> FineTune:
        """
        Submit RL tuning job to the platform.
        """
        # Resolve job name (fallback to class default)
        if job_name:
            self.tuning.name = job_name
        job_name_with_suffix = f"{self.tuning.name}-{generate_random_id()[:8]}"

        # Model name
        if model:
            self.tuning.model.name = model

        # rollouts/rewards
        if functions:
            self.tuning.functions = functions
        rollouts = self.tuning.combine_ids_runtimes(
            functype=FunctionType.ROLLOUT,
        )
        rewards = self.tuning.combine_ids_runtimes(
            functype=FunctionType.REWARD,
        )
        rewards.extend(
            self.tuning.combine_ids_runtimes(
                functype=FunctionType.GROUP_REWARD,
                id_str=get_func_type_id(FunctionType.REWARD),
            ),
        )
        # names of functions
        if not self.tuning.check_function_names():
            logger.error(
                "[%s] %s",
                AGENTIC_RL_DUPLICATE_FUNCTION_NAMES.name,
                AGENTIC_RL_DUPLICATE_FUNCTION_NAMES.format_message(),
            )
            exc = InvalidParameter(INVALID_REQUEST.format_msg())
            exc.status_code = INVALID_REQUEST.status_code
            exc.error_code = INVALID_REQUEST.error_code
            raise exc

        # datasets
        if datasets:
            self.tuning.datasets = datasets
        training_datasets = [
            ds
            for ds in self.tuning.datasets
            if ds.type == DatasetsType.TRAINING
        ]
        validation_datasets = [
            ds
            for ds in self.tuning.datasets
            if ds.type == DatasetsType.VALIDATION
        ]

        # hyper_parameters
        if hyper_parameters:
            self.tuning.training.hyper_parameters = hyper_parameters

        # resources
        if resources:
            self.tuning.training.resources = resources

        request = {
            "model": self.tuning.model.name,
            "training_datasets": [ds.model_dump() for ds in training_datasets],
            "validation_datasets": [
                ds.model_dump() for ds in validation_datasets
            ],
            "rollout": rollouts[0] if rollouts else None,
            "rewards": rewards,
            "hyper_parameters": self.tuning.training.hyper_parameters,
            "resource_config": self.tuning.training.resources,
            "training_type": str(self.tuning.training.type),
            "job_name": job_name_with_suffix,
        }
        request = deep_remove_none(request)
        logger.info(f"agentic_rl submit_job request: {request}")

        kwargs["base_address"] = DASHSCOPE_HTTP_BASE_URL
        try:
            resp = super().call(
                request,
                **kwargs,
            )
        except Exception as e:
            if isinstance(e, DashScopeException):
                raise
            logger.error(
                "[%s] %s | %s",
                AGENTIC_RL_JOB_SUBMISSION_FAILED.name,
                AGENTIC_RL_JOB_SUBMISSION_FAILED.format_message(),
                e,
                exc_info=True,
            )
            original_summary = f"{type(e).__name__}: {str(e)[:150]}"
            if _is_client_error(e):
                public_error = INVALID_REQUEST
                exc = InvalidParameter(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            else:
                public_error = INTERNAL_ERROR
                exc = DashScopeException(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            exc.status_code = public_error.status_code
            exc.error_code = public_error.error_code
            raise exc from e

        return FineTune(**resp)

    async def run(
        self,
        model: Optional[str] = None,
        # Datasets parameters
        training_datasets: Optional[List[TrainingDataset]] = None,
        validation_datasets: Optional[List[ValidationDataset]] = None,
        # Path-driven parameters (auto-register & upload)
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
        # Common parameters
        hyper_parameters: Optional[Dict[str, str]] = None,
        resources: Optional[Dict[str, str]] = None,
        job_name: Optional[str] = None,
        **kwargs,
    ) -> FineTune:
        """
        Execute RL tuning workflow.
        """
        try:
            logger.info(
                "Path-Driven mode: Registering functions & uploading "
                "datasets...",
            )
            await self.register_functions(
                functions=functions,
                lazy_load=True,
            )

            datasets = list(training_datasets or []) + list(
                validation_datasets or [],
            )
            await self.upload_datasets(
                datasets=datasets,
            )

            return self.submit_job(
                model=model,
                datasets=datasets,
                hyper_parameters=hyper_parameters,
                resources=resources,
                job_name=job_name,
                **kwargs,
            )
        except Exception as e:
            if isinstance(e, DashScopeException):
                raise
            logger.error(
                "[%s] %s | %s",
                AGENTIC_RL_WORKFLOW_FAILED.name,
                AGENTIC_RL_WORKFLOW_FAILED.format_message(),
                e,
                exc_info=True,
            )
            original_summary = f"{type(e).__name__}: {str(e)[:150]}"
            if _is_client_error(e):
                public_error = INVALID_REQUEST
                exc = InvalidParameter(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            else:
                public_error = INTERNAL_ERROR
                exc = DashScopeException(
                    f"{public_error.format_msg()} | "
                    f"Caused by: {original_summary}",
                )
            exc.status_code = public_error.status_code
            exc.error_code = public_error.error_code
            raise exc from e

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

    @classmethod
    async def test_functions(
        cls,
        instance_id: str,
        functype: FunctionType,
        input_data: Dict[str, Any],
        api_key: str = None,
    ):
        try:
            set_api_key(api_key)

            if functype == FunctionType.ROLLOUT:
                value = RolloutInput.model_validate(input_data)
            elif functype == FunctionType.REWARD:
                value = RewardInput.model_validate(input_data)
            elif functype == FunctionType.GROUP_REWARD:
                value = GroupRewardInput.model_validate(input_data)
            else:
                logger.error(
                    "[%s] %s | functype=%s",
                    AGENTIC_RL_UNSUPPORTED_FUNCTION_TYPE.name,
                    AGENTIC_RL_UNSUPPORTED_FUNCTION_TYPE.format_message(),
                    functype,
                )
                exc = InvalidParameter(INVALID_REQUEST.format_msg())
                exc.status_code = INVALID_REQUEST.status_code
                exc.error_code = INVALID_REQUEST.error_code
                raise exc

            logger.info(
                f"Starting {str(functype)} verification",
                extra={
                    "instance_id": instance_id,
                    "input_params": value.model_dump(exclude={"api_key"}),
                },
            )

            return await AgenticRLFunctionComponent.verify_function(
                value,
                instance_id,
            )

        except Exception as e:
            if isinstance(e, (DashScopeException, InvalidParameter)):
                raise

            original_summary = f"{type(e).__name__}: {str(e)[:150]}"

            if _is_client_error(e):
                # 4xx: client-side error (InputError, ValidationError, etc.)
                public_error = INVALID_REQUEST
                exc_class = InvalidParameter
                internal_error_def = AGENTIC_RL_FUNCTION_TEST_FAILED
            elif isinstance(e, (ConnectionError, OSError)):
                # 5xx: server/infra error
                public_error = INTERNAL_ERROR
                exc_class = DashScopeException
                internal_error_def = AGENTIC_RL_FUNCTION_TEST_FAILED
            elif isinstance(e, TimeoutError):
                # 5xx: timeout
                public_error = REQUEST_TIMEOUT
                exc_class = DashScopeException
                internal_error_def = AGENTIC_RL_FUNCTION_TEST_TIMEOUT
            else:
                # 5xx: validation / other → treat as client input issue
                public_error = INVALID_REQUEST
                exc_class = InvalidParameter
                internal_error_def = AGENTIC_RL_FUNCTION_TEST_FAILED

            logger.error(
                "[%s] %s | %s",
                internal_error_def.name,
                internal_error_def.format_message(),
                e,
                exc_info=True,
            )

            exc = exc_class(
                f"{public_error.format_msg()} | "
                f"Caused by: {original_summary}",
            )
            exc.status_code = public_error.status_code
            exc.error_code = public_error.error_code
            raise exc from e
