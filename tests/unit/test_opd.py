"""Minimal OPD contract tests."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.client.base_api import CreateMixin
from dashscope.finetune.agentic_rl import AgenticRL
from dashscope.finetune.reinforcement import (
    DataSourceType,
    RewardFunctionComponent,
    RolloutFunctionComponent,
    TrainingDataset,
    TrainingType,
)
from dashscope.finetune.reinforcement.common.errors import ValueErrorWithCode


@pytest.fixture
def rl_client():
    with patch("dashscope.finetune.agentic_rl.set_api_key"):
        return AgenticRL(api_key="test-key")


def _training_dataset():
    return TrainingDataset(
        data_source_type=DataSourceType.FILE_ID,
        file_id="file-train-001",
    )


def _success_response():
    return {
        "status_code": 200,
        "request_id": "request-001",
        "output": {
            "job_id": "ft-test-001",
            "status": "PENDING",
        },
    }


def test_pg_opd_training_type_is_public():
    assert TrainingType.PG_OPD == "pg_opd"


@pytest.mark.parametrize(
    ("functions", "has_rollout", "reward_count"),
    [
        (None, False, 0),
        ([RewardFunctionComponent(entity_id="reward-001")], False, 1),
        ([RolloutFunctionComponent(entity_id="rollout-001")], True, 0),
        (
            [
                RolloutFunctionComponent(entity_id="rollout-001"),
                RewardFunctionComponent(entity_id="reward-001"),
            ],
            True,
            1,
        ),
    ],
)
def test_teacher_model_enables_opd(
    rl_client,
    functions,
    has_rollout,
    reward_count,
):
    with patch.object(
        CreateMixin,
        "call",
        return_value=_success_response(),
    ) as call:
        rl_client.submit_job(
            model="qwen3.5-9b",
            datasets=[_training_dataset()],
            functions=functions,
            teacher_model="qwen3.5-397b-a17b",
        )

    request = call.call_args.args[0]
    assert request["teacher_model"] == "qwen3.5-397b-a17b"
    assert request["training_type"] == str(TrainingType.PG_OPD)
    assert ("rollout" in request) is has_rollout
    assert len(request.get("rewards", [])) == reward_count


def test_teacher_model_can_come_from_yaml(rl_client, tmp_path):
    config = tmp_path / "opd.yaml"
    config.write_text(
        "teacher_model: qwen3.5-397b-a17b\n",
        encoding="utf-8",
    )
    rl_client.init(config_path=str(config))

    with patch.object(
        CreateMixin,
        "call",
        return_value=_success_response(),
    ) as call:
        rl_client.submit_job(datasets=[_training_dataset()])

    request = call.call_args.args[0]
    assert request["teacher_model"] == "qwen3.5-397b-a17b"
    assert request["training_type"] == str(TrainingType.PG_OPD)


def test_opd_example_reuses_supported_rl_configuration(rl_client):
    config = (
        Path(__file__).parents[2]
        / "dashscope/finetune/reinforcement/examples/workspace"
        / "opd-job.yaml"
    )
    config_text = config.read_text(encoding="utf-8")
    rl_client.init(config_path=str(config))

    with patch.object(
        CreateMixin,
        "call",
        return_value=_success_response(),
    ) as call:
        rl_client.submit_job()

    request = call.call_args.args[0]
    assert request["teacher_model"] == "qwen3.5-397b-a17b"
    assert request["training_type"] == str(TrainingType.PG_OPD)
    assert rl_client.tuning.training.type == TrainingType.PG_OPD
    assert request["hyper_parameters"] == {
        "eval_steps": 1,
        "kl_loss_coef": 0.002,
        "learning_rate": "2e-6",
        "lr_scheduler_type": "cosine",
        "n_epochs": 1,
        "n_rollouts": 8,
        "ppo_mini_batch_size": 8,
    }
    assert rl_client.tuning.training.resources is None
    assert request["training_datasets"][0]["file_name"] == (
        "./data/calc_train_min.jsonl"
    )
    assert request["validation_datasets"][0]["file_name"] == (
        "./data/calc_validation_min.jsonl"
    )
    assert request["training_datasets"][0]["data_source_type"] == "file_id"
    assert "file_id" not in request["training_datasets"][0]
    assert [type(function) for function in rl_client.tuning.functions] == [
        RolloutFunctionComponent,
        RewardFunctionComponent,
    ]
    assert "Remove both function blocks: Teacher only." in config_text
    assert "Keep only the reward block: Teacher + Reward." in config_text
    assert "Keep only the rollout block: Teacher + Rollout." in config_text
    assert "Keep both blocks" in config_text


@pytest.mark.asyncio
async def test_opd_example_uploads_local_datasets(rl_client):
    config = (
        Path(__file__).parents[2]
        / "dashscope/finetune/reinforcement/examples/workspace"
        / "opd-job.yaml"
    )
    rl_client.init(config_path=str(config))

    with patch(
        "dashscope.finetune.reinforcement.common.model.to_bailian_data",
        new=AsyncMock(side_effect=[["file-train-001"], ["file-validation-001"]]),
    ):
        train_ids, validation_ids = await rl_client.upload_datasets()

    assert train_ids == ["file-train-001"]
    assert validation_ids == ["file-validation-001"]
    assert rl_client.tuning.datasets[0].file_id == "file-train-001"
    assert rl_client.tuning.datasets[1].file_id == "file-validation-001"


def test_regular_rl_example_keeps_resources_and_rl_hyperparameters(rl_client):
    config = (
        Path(__file__).parents[2]
        / "dashscope/finetune/reinforcement/examples/workspace"
        / "rl-job.yaml"
    )
    rl_client.init(config_path=str(config))

    assert rl_client.tuning.teacher_model is None
    assert rl_client.tuning.training.type == TrainingType.TRAINING_TYPE
    assert rl_client.tuning.training.hyper_parameters["algorithm"] == "gspo"
    assert rl_client.tuning.training.hyper_parameters["lr_scheduler_type"] == "linear"
    assert rl_client.tuning.training.resources == {
        "charge_type": "mtu_postpaid",
        "mtu_spec_code": "MTU4",
        "mtu_capacity": 24,
    }


def test_pg_opd_requires_teacher_model(rl_client):
    rl_client.tuning.training.type = TrainingType.PG_OPD

    with pytest.raises(
        ValueErrorWithCode,
        match=f"teacher_model is required when training.type is {TrainingType.PG_OPD}",
    ):
        rl_client.submit_job(datasets=[_training_dataset()])


def test_regular_reinforcement_is_unchanged(rl_client):
    with patch.object(
        CreateMixin,
        "call",
        return_value=_success_response(),
    ) as call:
        rl_client.submit_job(datasets=[_training_dataset()])

    request = call.call_args.args[0]
    assert request["training_type"] == "reinforcement"
    assert "teacher_model" not in request


@pytest.mark.asyncio
async def test_run_forwards_teacher_model(rl_client):
    submit = MagicMock(return_value="submitted")

    with (
        patch.object(
            AgenticRL,
            "register_functions",
            new=AsyncMock(),
        ),
        patch.object(
            AgenticRL,
            "upload_datasets",
            new=AsyncMock(),
        ),
        patch.object(
            AgenticRL,
            "submit_job",
            new=submit,
        ),
    ):
        result = await rl_client.run(teacher_model="qwen3.5-397b-a17b")

    assert result == "submitted"
    assert submit.call_args.kwargs["teacher_model"] == "qwen3.5-397b-a17b"
