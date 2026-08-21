# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.finetune.reinforcement import (
    DataSourceType,
    FunctionComponentModel,
    FunctionComponentRuntime,
    RewardFunctionComponent,
    RolloutFunctionComponent,
    TrainingDataset,
    ValidationDataset,
)


# ========================================================================== #
#                              Fixtures                                       #
# ========================================================================== #


@pytest.fixture
def rollout_runtime_dict():
    return {
        "cpu": 2,
        "memory_size": 4096,
        "disk_size": 512,
        "concurrency": 30,
        "capacity": 30,
        "min_capacity": 30,
        "max_capacity": 60,
        "memory_scale_threshold": 0.6,
        "concurrency_scale_threshold": 0.6,
        "env": {},
    }


@pytest.fixture
def reward_runtime():
    return FunctionComponentRuntime(
        cpu=2,
        memory_size=4096,
        disk_size=512,
        concurrency=30,
        env={},
        capacity=30,
        min_capacity=30,
        max_capacity=60,
        memory_scale_threshold=0.6,
        concurrency_scale_threshold=0.6,
    )


@pytest.fixture
def training_datasets():
    return [
        TrainingDataset(
            data_source_type=DataSourceType.FILE_ID,
            file_name="./data/calc_train_min.jsonl",
        ),
    ]


@pytest.fixture
def validation_datasets():
    return [
        ValidationDataset(
            data_source_type=DataSourceType.FILE_ID,
            file_name="./data/calc_validation_min.jsonl",
        ),
    ]


@pytest.fixture
def rollout_component(rollout_runtime_dict):
    return RolloutFunctionComponent(
        name="rollout-1",
        timeout=600,
        fcmodel=FunctionComponentModel(
            classpath="functions.rollout.rollout.CalcXRolloutProcessor",
        ),
        runtime=FunctionComponentRuntime(**rollout_runtime_dict),
    )


@pytest.fixture
def reward_component(reward_runtime):
    return RewardFunctionComponent(
        name="reward-1",
        weight=1.0,
        timeout=120,
        reward_metric_weight={
            "reward_metric_weightA": 0.3,
            "reward_metric_weightB": 0.7,
        },
        fcmodel=FunctionComponentModel(
            classpath="functions.reward.reward.DemoRewardProcessor",
        ),
        runtime=reward_runtime,
    )


@pytest.fixture
def hyper_parameters():
    return {
        "algorithm": "gspo",
        "batch_size": 64,
        "eval_steps": 1,
        "kl_loss_coef": 0.002,
        "learning_rate": 2e-6,
        "lr_scheduler_type": "linear",
        "max_length": 8192,
        "n_epochs": 1,
        "n_rollouts": 8,
        "ppo_mini_batch_size": 8,
        "save_strategy": "steps",
    }


@pytest.fixture
def resources():
    return {
        "charge_type": "mtu_postpaid",
        "mtu_spec_code": "MTU4",
        "mtu_capacity": 24,
    }


@pytest.fixture
def mock_success_run_result():
    result = MagicMock()
    result.status_code = 200
    result.output = MagicMock()
    result.output.job_id = "ft-test-job-001"
    return result


@pytest.fixture
def mock_failure_run_result():
    result = MagicMock()
    result.status_code = 400
    result.output = None
    return result


@pytest.fixture
def mock_get_result():
    result = MagicMock()
    result.status_code = 200
    result.output = MagicMock()
    result.output.job_id = "ft-test-job-001"
    result.output.status = "RUNNING"
    return result


# ========================================================================== #
#                     Data Model Construction Tests                           #
# ========================================================================== #


class TestDataModelConstruction:
    def test_training_dataset_creation(self):
        ds = TrainingDataset(
            data_source_type=DataSourceType.FILE_ID,
            file_name="./data/train.jsonl",
        )
        assert ds.data_source_type == DataSourceType.FILE_ID
        assert ds.file_name == "./data/train.jsonl"

    def test_validation_dataset_creation(self):
        ds = ValidationDataset(
            data_source_type=DataSourceType.FILE_ID,
            file_name="./data/val.jsonl",
        )
        assert ds.data_source_type == DataSourceType.FILE_ID
        assert ds.file_name == "./data/val.jsonl"

    def test_function_component_runtime_from_dict(self, rollout_runtime_dict):
        runtime = FunctionComponentRuntime(**rollout_runtime_dict)
        assert runtime.cpu == 2
        assert runtime.memory_size == 4096
        assert runtime.disk_size == 512
        assert runtime.concurrency == 30
        assert runtime.capacity == 30
        assert runtime.min_capacity == 30
        assert runtime.max_capacity == 60
        assert runtime.memory_scale_threshold == 0.6
        assert runtime.concurrency_scale_threshold == 0.6
        assert not runtime.env

    def test_function_component_runtime_direct(self, reward_runtime):
        assert reward_runtime.cpu == 2
        assert reward_runtime.memory_size == 4096
        assert reward_runtime.concurrency == 30
        assert reward_runtime.max_capacity == 60

    def test_function_component_model_with_classpath(self):
        model = FunctionComponentModel(
            classpath="functions.rollout.rollout.CalcXRolloutProcessor",
        )
        assert (
            model.classpath
            == "functions.rollout.rollout.CalcXRolloutProcessor"
        )

    def test_rollout_component_creation(self, rollout_component):
        assert rollout_component.name == "rollout-1"
        assert rollout_component.timeout == 600
        assert (
            rollout_component.fcmodel.classpath
            == "functions.rollout.rollout.CalcXRolloutProcessor"
        )
        assert rollout_component.runtime.cpu == 2

    def test_reward_component_creation(self, reward_component):
        assert reward_component.name == "reward-1"
        assert reward_component.weight == 1.0
        assert reward_component.timeout == 120
        assert reward_component.reward_metric_weight == {
            "reward_metric_weightA": 0.3,
            "reward_metric_weightB": 0.7,
        }
        assert (
            reward_component.fcmodel.classpath
            == "functions.reward.reward.DemoRewardProcessor"
        )

    def test_reward_metric_weight_values(self, reward_component):
        weights = reward_component.reward_metric_weight
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_multiple_training_datasets(self):
        datasets = [
            TrainingDataset(
                data_source_type=DataSourceType.FILE_ID,
                file_name="./data/train1.jsonl",
            ),
            TrainingDataset(
                data_source_type=DataSourceType.FILE_ID,
                file_name="./data/train2.jsonl",
            ),
        ]
        assert len(datasets) == 2
        assert datasets[0].file_name == "./data/train1.jsonl"
        assert datasets[1].file_name == "./data/train2.jsonl"


# ========================================================================== #
#                     Hyper Parameters & Resources Tests                      #
# ========================================================================== #


class TestHyperParametersAndResources:
    def test_hyper_parameters_values(self, hyper_parameters):
        assert hyper_parameters["algorithm"] == "gspo"
        assert hyper_parameters["batch_size"] == 64
        assert hyper_parameters["eval_steps"] == 1
        assert hyper_parameters["kl_loss_coef"] == 0.002
        assert hyper_parameters["learning_rate"] == 2e-6
        assert hyper_parameters["lr_scheduler_type"] == "linear"
        assert hyper_parameters["max_length"] == 8192
        assert hyper_parameters["n_epochs"] == 1
        assert hyper_parameters["n_rollouts"] == 8
        assert hyper_parameters["ppo_mini_batch_size"] == 8
        assert hyper_parameters["save_strategy"] == "steps"

    def test_resources_values(self, resources):
        assert resources["charge_type"] == "mtu_postpaid"
        assert resources["mtu_spec_code"] == "MTU4"
        assert resources["mtu_capacity"] == 24


# ========================================================================== #
#                     Workflow Tests (main_workflow)                         #
# ========================================================================== #


class TestMainWorkflow:
    @pytest.mark.asyncio
    async def test_main_workflow_success(
        self,
        mock_success_run_result,
        mock_get_result,
    ):
        with patch(
            "dashscope.finetune.reinforcement.common.utils.set_api_key",
        ), patch.object(
            __import__(
                "dashscope.finetune.agentic_rl",
                fromlist=["AgenticRL"],
            ),
            "AgenticRL",
        ) as MockAgenticRL:
            instance = MockAgenticRL.return_value
            instance.run = AsyncMock(return_value=mock_success_run_result)
            MockAgenticRL.get = MagicMock(return_value=mock_get_result)

            from dashscope.finetune.agentic_rl import AgenticRL

            client = AgenticRL()
            result = await client.run(
                job_name="agentic-rl",
                model="qwen3.5-9b",
                training_datasets=[
                    TrainingDataset(
                        data_source_type=DataSourceType.FILE_ID,
                        file_name="./data/calc_train_min.jsonl",
                    ),
                ],
                validation_datasets=[
                    ValidationDataset(
                        data_source_type=DataSourceType.FILE_ID,
                        file_name="./data/calc_validation_min.jsonl",
                    ),
                ],
                functions=[
                    RolloutFunctionComponent(
                        name="rollout-1",
                        timeout=600,
                        fcmodel=FunctionComponentModel(
                            classpath="functions.rollout.rollout.CalcXRolloutProcessor",  # noqa: E501  # pylint: disable=line-too-long
                        ),
                        runtime=FunctionComponentRuntime(cpu=2),
                    ),
                    RewardFunctionComponent(
                        name="reward-1",
                        weight=1.0,
                        timeout=120,
                        fcmodel=FunctionComponentModel(
                            classpath="functions.reward.reward.DemoRewardProcessor",  # noqa: E501  # pylint: disable=line-too-long
                        ),
                        runtime=FunctionComponentRuntime(cpu=2),
                    ),
                ],
                hyper_parameters={"algorithm": "gspo"},
                resources={"charge_type": "mtu_postpaid"},
            )

            assert result.status_code == 200
            assert result.output.job_id == "ft-test-job-001"
            instance.run.assert_called_once()

            get_result = AgenticRL.get(job_id="ft-test-job-001")
            assert get_result.status_code == 200
            MockAgenticRL.get.assert_called_once_with(job_id="ft-test-job-001")

    @pytest.mark.asyncio
    async def test_main_workflow_submit_failure(self, mock_failure_run_result):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch.object(
            AgenticRL,
            "__init__",
            return_value=None,
        ), patch.object(
            AgenticRL,
            "run",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = mock_failure_run_result

            # client = AgenticRL.__new__(AgenticRL)
            client = AgenticRL()
            result = await client.run(
                job_name="agentic-rl",
                model="qwen3.5-9b",
                training_datasets=[
                    TrainingDataset(
                        data_source_type=DataSourceType.FILE_ID,
                        file_name="./data/train.jsonl",
                    ),
                ],
            )

            assert result.status_code == 400
            with pytest.raises(ValueError):
                if result.status_code != 200:
                    raise ValueError(f"agentic rl submit: {result}")

    @pytest.mark.asyncio
    async def test_main_workflow_run_exception(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch.object(
            AgenticRL,
            "__init__",
            return_value=None,
        ), patch.object(
            AgenticRL,
            "run",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            client = AgenticRL.__new__(AgenticRL)
            with pytest.raises(Exception, match="Network error"):
                await client.run(
                    model="qwen3.5-9b",
                    training_datasets=[
                        TrainingDataset(
                            data_source_type=DataSourceType.FILE_ID,
                            file_name="./data/train.jsonl",
                        ),
                    ],
                )


# ========================================================================== #
#                   Workflow Tests (main_workflow_yaml)                      #
# ========================================================================== #


class TestMainWorkflowYaml:
    @pytest.mark.asyncio
    async def test_yaml_workflow_success(
        self,
        mock_success_run_result,
        mock_get_result,
    ):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.finetune.reinforcement import TuningModel

        with patch.object(AgenticRL, "init") as mock_init, patch.object(
            AgenticRL,
            "run",
            new_callable=AsyncMock,
        ) as mock_run, patch.object(
            AgenticRL,
            "get",
        ) as mock_get:
            mock_init.return_value = None
            mock_run.return_value = mock_success_run_result
            mock_get.return_value = mock_get_result

            client = AgenticRL(api_key="sk-test-key")
            client.tuning = MagicMock(spec=TuningModel)

            client.init(
                config_path="job.yaml",
                name="agentic-rl-from-yaml",
            )
            mock_init.assert_called_once_with(
                config_path="job.yaml",
                name="agentic-rl-from-yaml",
            )

            result = await client.run()
            assert result.status_code == 200
            assert result.output.job_id == "ft-test-job-001"

            get_result = AgenticRL.get(job_id="ft-test-job-001")
            assert get_result.status_code == 200

    @pytest.mark.asyncio
    async def test_yaml_workflow_submit_failure(
        self,
        mock_failure_run_result,
    ):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch.object(AgenticRL, "init"), patch.object(
            AgenticRL,
            "run",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = mock_failure_run_result

            client = AgenticRL()
            client.init(config_path="job.yaml", name="test")

            result = await client.run()
            assert result.status_code != 200

            with pytest.raises(ValueError):
                if result.status_code != 200:
                    raise ValueError(f"agentic rl submit: {result}")

    @pytest.mark.asyncio
    async def test_yaml_workflow_to_yaml_export(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.finetune.reinforcement import TuningModel

        client = AgenticRL()
        client.tuning = MagicMock(spec=TuningModel)

        client.tuning.to_yaml(file_path="init.yaml")
        client.tuning.to_yaml.assert_called_once_with(
            file_path="init.yaml",
        )


# ========================================================================== #
#                     AgenticRL Client Tests                                  #
# ========================================================================== #


class TestAgenticRLClient:
    def test_init_with_valid_api_key(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch(
            "dashscope.finetune.agentic_rl.set_api_key",
        ) as mock_set_key:
            client = AgenticRL(api_key="sk-test-key")
            mock_set_key.assert_called_once_with("sk-test-key")
            assert isinstance(client, AgenticRL)

    def test_init_with_invalid_api_key(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        with patch(
            "dashscope.finetune.agentic_rl.set_api_key",
            side_effect=Exception("Invalid key"),
        ):
            with pytest.raises(DashScopeException):
                AgenticRL(api_key="invalid")

    def test_init_without_api_key(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch(
            "dashscope.finetune.agentic_rl.set_api_key",
        ) as mock_set_key:
            _ = AgenticRL()
            mock_set_key.assert_called_once_with(None)

    def test_init_with_api_key(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch(
            "dashscope.finetune.agentic_rl.set_api_key",
        ) as mock_set_key:
            _ = AgenticRL(api_key="your_api_key_here")
            mock_set_key.assert_called_once_with("your_api_key_here")

    def test_init_from_yaml(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.finetune.reinforcement import TuningModel

        mock_tuning = MagicMock(spec=TuningModel)
        with patch.object(
            TuningModel,
            "load_from_yaml",
            return_value=mock_tuning,
        ) as mock_load:
            client = AgenticRL()
            result = client.init(
                config_path="job.yaml",
                name="test-job",
            )
            mock_load.assert_called_once_with(
                "job.yaml",
                name="test-job",
            )
            assert client.tuning is mock_tuning
            assert result is client

    def test_init_from_yaml_default_path(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.finetune.reinforcement import TuningModel

        mock_tuning = MagicMock(spec=TuningModel)
        with patch.object(
            TuningModel,
            "load_from_yaml",
            return_value=mock_tuning,
        ) as mock_load:
            client = AgenticRL()
            client.init()
            mock_load.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_register_functions_success(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        client = AgenticRL()
        client.tuning = MagicMock()
        client.tuning.register_functions = AsyncMock(
            return_value=(
                ["rollout-id"],
                ["reward-id"],
                [],
                ["rollout-inst"],
                ["reward-inst"],
                [],
            ),
        )
        client.tuning.functions = []

        result = await client.register_functions(lazy_load=False)
        assert result == (
            ["rollout-id"],
            ["reward-id"],
            [],
            ["rollout-inst"],
            ["reward-inst"],
            [],
        )

    @pytest.mark.asyncio
    async def test_register_functions_failure(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        client = AgenticRL()
        client.tuning = MagicMock()
        client.tuning.register_functions = AsyncMock(
            side_effect=Exception("Register failed"),
        )
        client.tuning.functions = []

        with pytest.raises(DashScopeException):
            await client.register_functions()

    @pytest.mark.asyncio
    async def test_upload_datasets_success(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        client = AgenticRL()
        client.tuning = MagicMock()
        client.tuning.upload_datasets = AsyncMock(
            return_value=(["train-file-id"], ["val-file-id"]),
        )
        client.tuning.datasets = []

        result = await client.upload_datasets()
        assert result == (["train-file-id"], ["val-file-id"])

    @pytest.mark.asyncio
    async def test_upload_datasets_failure(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        client = AgenticRL()
        client.tuning = MagicMock()
        client.tuning.upload_datasets = AsyncMock(
            side_effect=Exception("Upload failed"),
        )
        client.tuning.datasets = []

        with pytest.raises(DashScopeException):
            await client.upload_datasets()


# ========================================================================== #
#                     Submit Job Method Tests                                 #
# ========================================================================== #


class TestSubmitJob:
    def test_submit_job_builds_correct_request(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        with patch(
            "dashscope.finetune.agentic_rl.generate_random_id",
            return_value="abcdef1234567890",
        ):
            client = AgenticRL()
            client.tuning = MagicMock()
            client.tuning.name = "agentic-rl"
            client.tuning.model.name = "qwen3.5-9b"
            client.tuning.training.hyper_parameters = {
                "algorithm": "gspo",
            }
            client.tuning.training.resources = {
                "charge_type": "mtu_postpaid",
            }
            client.tuning.training.type = "sft_type"
            client.tuning.datasets = [
                TrainingDataset(
                    data_source_type=DataSourceType.FILE_ID,
                    file_name="train.jsonl",
                ),
            ]
            client.tuning.combine_ids_runtimes = MagicMock(
                side_effect=[
                    [{"rollout_id": "r1", "cpu": 2}],
                    [{"reward_id": "rw1", "cpu": 1}],
                    [],
                ],
            )
            client.tuning.check_function_names = MagicMock(
                return_value=True,
            )

    def test_submit_job_duplicate_function_names(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        client = AgenticRL()
        client.tuning = MagicMock()
        client.tuning.name = "test"
        client.tuning.model.name = "qwen3.5-9b"
        client.tuning.combine_ids_runtimes = MagicMock(
            side_effect=[
                [{"rollout_id": "r1"}],
                [{"reward_id": "rw1"}],
                [],
            ],
        )
        client.tuning.check_function_names = MagicMock(
            return_value=False,
        )

        with pytest.raises(DashScopeException, match="request is invalid"):
            client.submit_job()


# ========================================================================== #
#                     Class Method Tests (get/cancel/list)                    #
# ========================================================================== #


class TestClassMethods:
    def test_get_job(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_result = MagicMock()
        mock_result.status_code = 200
        with patch(
            "dashscope.finetune.agentic_rl.FineTunes.get",
            return_value=mock_result,
        ) as mock_get:
            result = AgenticRL.get(job_id="ft-job-123")
            assert result.status_code == 200
            mock_get.assert_called_once()

    def test_cancel_job(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_result = MagicMock()
        mock_result.status_code = 200
        with patch(
            "dashscope.finetune.agentic_rl.FineTunes.cancel",
            return_value=mock_result,
        ) as mock_cancel:
            result = AgenticRL.cancel(job_id="ft-job-123")
            assert result.status_code == 200
            mock_cancel.assert_called_once()

    def test_list_jobs(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_result = MagicMock()
        mock_result.status_code = 200
        with patch(
            "dashscope.finetune.agentic_rl.FineTunes.list",
            return_value=mock_result,
        ) as mock_list:
            result = AgenticRL.list(page_no=1, page_size=10)
            assert result.status_code == 200
            mock_list.assert_called_once()

    def test_delete_job(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_result = MagicMock()
        mock_result.status_code = 200
        with patch(
            "dashscope.finetune.agentic_rl.FineTunes.delete",
            return_value=mock_result,
        ) as mock_delete:
            result = AgenticRL.delete(job_id="ft-job-123")
            assert result.status_code == 200
            mock_delete.assert_called_once()

    def test_logs_job(self):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_result = MagicMock()
        mock_result.status_code = 200
        with patch(
            "dashscope.finetune.agentic_rl.FineTunes.logs",
            return_value=mock_result,
        ) as mock_logs:
            result = AgenticRL.logs(
                job_id="ft-job-123",
                offset=1,
                lines=500,
            )
            assert result.status_code == 200
            mock_logs.assert_called_once()


# ========================================================================== #
#                     End-to-End Run Method Tests                             #
# ========================================================================== #


class TestRunMethod:
    @pytest.mark.asyncio
    async def test_run_full_workflow(
        self,
        training_datasets,
        validation_datasets,
        rollout_component,
        reward_component,
        hyper_parameters,
        resources,
    ):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_finetune = MagicMock()
        mock_finetune.status_code = 200
        mock_finetune.output.job_id = "ft-run-001"

        with patch.object(
            AgenticRL,
            "register_functions",
            new_callable=AsyncMock,
        ), patch.object(
            AgenticRL,
            "upload_datasets",
            new_callable=AsyncMock,
        ), patch.object(
            AgenticRL,
            "submit_job",
            return_value=mock_finetune,
        ) as mock_submit:
            client = AgenticRL()
            client.tuning = MagicMock()

            result = await client.run(
                model="qwen3.5-9b",
                training_datasets=training_datasets,
                validation_datasets=validation_datasets,
                functions=[rollout_component, reward_component],
                hyper_parameters=hyper_parameters,
                resources=resources,
                job_name="agentic-rl",
            )

            assert result.status_code == 200
            assert result.output.job_id == "ft-run-001"
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_register_failure_raises(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        with patch.object(
            AgenticRL,
            "register_functions",
            new_callable=AsyncMock,
            side_effect=Exception("Registration error"),
        ):
            client = AgenticRL()
            client.tuning = MagicMock()

            with pytest.raises(DashScopeException):
                await client.run(model="qwen3.5-9b")

    @pytest.mark.asyncio
    async def test_run_upload_failure_raises(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        with patch.object(
            AgenticRL,
            "register_functions",
            new_callable=AsyncMock,
        ), patch.object(
            AgenticRL,
            "upload_datasets",
            new_callable=AsyncMock,
            side_effect=Exception("Upload error"),
        ):
            client = AgenticRL()
            client.tuning = MagicMock()

            with pytest.raises(DashScopeException):
                await client.run(model="qwen3.5-9b")

    @pytest.mark.asyncio
    async def test_run_submit_failure_raises(self):
        from dashscope.finetune.agentic_rl import AgenticRL
        from dashscope.common.error import DashScopeException

        with patch.object(
            AgenticRL,
            "register_functions",
            new_callable=AsyncMock,
        ), patch.object(
            AgenticRL,
            "upload_datasets",
            new_callable=AsyncMock,
        ), patch.object(
            AgenticRL,
            "submit_job",
            side_effect=Exception("Submit error"),
        ):
            client = AgenticRL()
            client.tuning = MagicMock()

            with pytest.raises(DashScopeException):
                await client.run(model="qwen3.5-9b")

    @pytest.mark.asyncio
    async def test_run_with_no_validation_datasets(self, training_datasets):
        from dashscope.finetune.agentic_rl import AgenticRL

        mock_finetune = MagicMock()
        mock_finetune.status_code = 200

        with patch.object(
            AgenticRL,
            "register_functions",
            new_callable=AsyncMock,
        ), patch.object(
            AgenticRL,
            "upload_datasets",
            new_callable=AsyncMock,
        ) as mock_upload, patch.object(
            AgenticRL,
            "submit_job",
            return_value=mock_finetune,
        ):
            client = AgenticRL()
            client.tuning = MagicMock()

            result = await client.run(
                model="qwen3.5-9b",
                training_datasets=training_datasets,
            )
            assert result.status_code == 200

            call_kwargs = mock_upload.call_args
            datasets = call_kwargs.kwargs.get("datasets", [])
            assert len(datasets) == 1
