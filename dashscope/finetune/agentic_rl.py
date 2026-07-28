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
from dashscope.finetune.reinforcement.common.errors import (
    RegistrationError,
    ValidationError,
    RuntimeErrorWithCode,
    ValueErrorWithCode,
    DatasetsError,
)
from dashscope.common.error_registry import (
    INVALID_API_KEY,
    INVALID_REQUEST,
    INTERNAL_ERROR,
)


def _convert_to_public_exception(exc: Exception) -> Exception:
    """Convert internal AgenticRL exceptions to public SDK exceptions.

    Follows the error.md principle: API-returned errors (DashScopeException
    and its subclasses) are passed through unchanged to preserve the original
    code, message, status_code, and request_id from the HTTP API response.

    For internal AgenticRL exceptions with a public_error mapping, converts
    them to standard SDK exceptions via to_public_exception(). Otherwise logs
    the internal error and returns a generic InternalServerError to prevent
    leaking internal error codes.
    """
    # Pass through API-returned standard exceptions unchanged
    from dashscope.common.error import DashScopeException

    if isinstance(exc, DashScopeException):
        return exc

    if hasattr(exc, "to_public_exception"):
        converted = exc.to_public_exception()
        # If to_public_exception returned self (no public_error set),
        # fall through to the safe fallback below
        if converted is not exc:
            return converted

    # Safe fallback: log internal details, never expose integer error codes
    internal_code = getattr(exc, "error_code", None)
    logger.error(
        "AgenticRL internal error (no public_error mapping): code=%s, "
        "type=%s, message=%s",
        internal_code,
        type(exc).__name__,
        str(exc),
        exc_info=True,
    )

    formatted_msg = INTERNAL_ERROR.format_msg()
    safe_exc = DashScopeException(formatted_msg)
    safe_exc.status_code = INTERNAL_ERROR.status_code
    safe_exc.error_code = INTERNAL_ERROR.error_code
    safe_exc.request_id = None  # type: ignore[attr-defined]
    return safe_exc


class AgenticRL(AgenticRLTuning, CreateMixin):
    SUB_PATH: ClassVar[str] = "fine-tunes"

    def __init__(self, api_key: str = None):
        super().__init__()

        try:
            set_api_key(api_key)
        except Exception as e:
            raise _convert_to_public_exception(
                ValueErrorWithCode(
                    "Invalid API key configuration",
                    error_code=3001,
                    public_error=INVALID_API_KEY,
                ),
            ) from e

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
            if hasattr(e, "error_code"):
                raise _convert_to_public_exception(e) from e
            raise _convert_to_public_exception(
                RegistrationError(
                    "Function registration failed",
                    error_code=3002,
                    public_error=INTERNAL_ERROR,
                ),
            ) from e

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
            raise _convert_to_public_exception(
                DatasetsError(
                    "Datasets upload failed",
                    error_code=3003,
                    public_error=INTERNAL_ERROR,
                ),
            ) from e

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
            raise _convert_to_public_exception(
                ValueErrorWithCode(
                    "Duplicate function names detected. "
                    "All function names must be unique.",
                    error_code=3004,
                    public_error=INVALID_REQUEST,
                ),
            )

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
            if hasattr(e, "error_code"):
                raise _convert_to_public_exception(e) from e
            raise _convert_to_public_exception(
                RuntimeErrorWithCode(
                    "Job submission failed",
                    error_code=3005,
                    public_error=INTERNAL_ERROR,
                ),
            ) from e

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
            if hasattr(e, "error_code"):
                raise _convert_to_public_exception(e) from e
            raise _convert_to_public_exception(
                RuntimeErrorWithCode(
                    "RL tuning workflow failed",
                    error_code=3006,
                    public_error=INTERNAL_ERROR,
                ),
            ) from e

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
                raise _convert_to_public_exception(
                    ValueErrorWithCode(
                        f"Unsupported function type: {functype}",
                        error_code=3007,
                        public_error=INVALID_REQUEST,
                    ),
                )

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
            raise _convert_to_public_exception(
                ValidationError(
                    "Function test failed",
                    error_code=3008,
                    public_error=INVALID_REQUEST,
                ),
            ) from e
