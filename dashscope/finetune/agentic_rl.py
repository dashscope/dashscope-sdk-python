# -*- coding: utf-8 -*-
from __future__ import annotations

# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Union, List, Optional, ClassVar, Dict, Any
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
    InstanceQueryError,
    RuntimeErrorWithCode,
    ValueErrorWithCode,
    DatasetsError,
)


class AgenticRL(AgenticRLTuning, CreateMixin):
    SUB_PATH: ClassVar[str] = "fine-tunes"

    def __init__(self, api_key: str = None):
        super().__init__()

        try:
            set_api_key(api_key)
        except Exception as e:
            raise ValueErrorWithCode(
                "Invalid API key configuration",
                error_code=3001,
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
    ) -> tuple[
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
                raise
            raise RegistrationError(
                "Function registration failed",
                error_code=3002,
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
    ) -> tuple[List[str], List[str]]:
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
            raise DatasetsError(
                "Datasets upload failed",
                error_code=3003,
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
            raise ValueErrorWithCode(
                "Duplicate function names detected. All function names must "
                "be unique.",
                error_code=3004,
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
                raise
            raise RuntimeErrorWithCode(
                "Job submission failed",
                error_code=3005,
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
                raise
            raise RuntimeErrorWithCode(
                "RL tuning workflow failed",
                error_code=3006,
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
        pull_logs: bool = False,
        log_page_size: int = 100,
    ):
        """Test a deployed function instance with custom input data.

        Args:
            instance_id: Target function instance ID.
            functype: Function type (ROLLOUT/REWARD/GROUP_REWARD).
            input_data: Test input payload.
            api_key: DashScope API key (uses DASHSCOPE_API_KEY env var
                if omitted).
            pull_logs: If True, pull all logs of the function instance
                (with pagination) after verification and print them
                between separator markers.
            log_page_size: Page size used when pulling logs.
        """
        try:
            set_api_key(api_key)

            if functype == FunctionType.ROLLOUT:
                value = RolloutInput.model_validate(input_data)
            elif functype == FunctionType.REWARD:
                value = RewardInput.model_validate(input_data)
            elif functype == FunctionType.GROUP_REWARD:
                value = GroupRewardInput.model_validate(input_data)
            else:
                raise ValueErrorWithCode(
                    f"Unsupported function type: {functype}",
                    error_code=3007,
                )

            logger.info(
                f"Starting {str(functype)} verification",
                extra={
                    "instance_id": instance_id,
                    "input_params": value.model_dump(exclude={"api_key"}),
                },
            )

            result = await AgenticRLFunctionComponent.verify_function(
                value,
                instance_id,
            )

            if pull_logs:
                await cls._pull_function_instance_logs(
                    function_instance_id=instance_id,
                    page_size=log_page_size,
                )

            return result

        except Exception as e:
            raise ValidationError(
                "Function test failed",
                error_code=3008,
            ) from e

    @classmethod
    async def query_function_instance_logs(
        cls,
        function_instance_id: str,
        page_number: int = 1,
        page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        api_key: str = None,
    ) -> Dict[str, Any]:
        """Query one page of logs for a function (faas) instance.

        Args:
            function_instance_id: Target function instance ID.
            page_number: Page number, starting from 1.
            page_size: Number of log entries per page.
            start_time: Optional start time filter (in seconds).
            end_time: Optional end time filter (in seconds).
            keywords: Optional keyword filters for log messages.
            api_key: DashScope API key (uses DASHSCOPE_API_KEY env var
                if omitted).

        Returns:
            Raw response dict of the log query API.
        """
        try:
            set_api_key(api_key)

            fc_component = AgenticRLFunctionComponent
            return await fc_component.query_function_instance_logs(
                function_instance_id=function_instance_id,
                page_number=page_number,
                page_size=page_size,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
            )

        except Exception as e:
            if hasattr(e, "error_code"):
                raise
            raise InstanceQueryError(
                "Function instance log query failed",
                error_code=3009,
            ) from e

    @classmethod
    async def query_all_function_instance_logs(
        cls,
        function_instance_id: str,
        page_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        api_key: str = None,
    ) -> List[str]:
        """Fetch all logs of a function (faas) instance.

        Pages are fetched one by one via the paginated log query until
        all entries are collected.

        Args:
            function_instance_id: Target function instance ID.
            page_size: Number of log entries per page.
            start_time: Optional start time filter (in seconds).
            end_time: Optional end time filter (in seconds).
            keywords: Optional keyword filters for log messages.
            api_key: DashScope API key (uses DASHSCOPE_API_KEY env var
                if omitted).

        Returns:
            Aggregated list of log messages across all pages.
        """
        try:
            set_api_key(api_key)

            fc_component = AgenticRLFunctionComponent
            return await fc_component.query_all_function_instance_logs(
                function_instance_id=function_instance_id,
                page_size=page_size,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
            )

        except Exception as e:
            if hasattr(e, "error_code"):
                raise
            raise InstanceQueryError(
                "Function instance log query failed",
                error_code=3010,
            ) from e

    @classmethod
    async def _pull_function_instance_logs(
        cls,
        function_instance_id: str,
        page_size: int = 100,
    ) -> None:
        """Pull all logs of a function instance and print them between
        separator markers (best effort, never raises)."""
        logger.info(
            "************start query log "
            f"(function_instance_id={function_instance_id})*******",
        )
        try:
            logs = await cls.query_all_function_instance_logs(
                function_instance_id=function_instance_id,
                page_size=page_size,
            )
            for entry in logs:
                logger.info(f"[function instance log] {entry}")
            logger.info(
                f"Pulled {len(logs)} log entries for function instance "
                f"{function_instance_id}",
            )
        except Exception as e:
            logger.warning(
                f"Failed to pull logs for function instance "
                f"{function_instance_id}: {e}",
            )
        finally:
            logger.info("************end query log *******")
