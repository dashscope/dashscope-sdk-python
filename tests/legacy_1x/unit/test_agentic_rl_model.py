# -*- coding: utf-8 -*-
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from dashscope.finetune.reinforcement import (
    AgenticRLTuning,
    TuningModel,
    RolloutFunctionComponent,
    RewardFunctionComponent,
    FunctionType,
    TrainingType,
    DataSourceType,
    TrainingDataset,
    ValidationDataset,
    FoundationModel,
    Training,
    FunctionComponentModel,
    FunctionComponentRuntime,
    RegistrationError,
    OSSUploadError,
    get_filepath_classname,
)


@pytest.fixture
def sample_tuning_model():
    return TuningModel(
        name="test-tuning",
        datasets=[
            TrainingDataset(
                data_source_type=DataSourceType.FILE_ID,
                file_name="train1.jsonl",
            ),
            TrainingDataset(
                data_source_type=DataSourceType.FILE_ID,
                file_name="train2.jsonl",
            ),
            ValidationDataset(
                data_source_type=DataSourceType.FILE_ID,
                file_name="val.jsonl",
            ),
        ],
        model=FoundationModel(name="qwen-max"),
        training=Training(
            type=TrainingType.TRAINING_TYPE,
            hyper_parameters={"learning_rate": "1e-4"},
        ),
    )


@pytest.fixture
def sample_function_components():
    return [
        RolloutFunctionComponent(
            type=FunctionType.ROLLOUT,
            fcmodel=FunctionComponentModel(
                zipdir="./",
                filepath="functions/rollout.py",
                classname="RolloutProcessor",
            ),
            runtime=FunctionComponentRuntime(
                cpu=2,
                memory_size=4096,
                concurrency=10,
            ),
        ),
        RewardFunctionComponent(
            type=FunctionType.REWARD,
            fcmodel=FunctionComponentModel(
                zipdir="./",
                filepath="functions/reward.py",
                classname="RewardProcessor",
            ),
            runtime=FunctionComponentRuntime(
                cpu=1,
                memory_size=4096,
                concurrency=5,
            ),
        ),
    ]


# pylint: disable=redefined-outer-name
@pytest.fixture
def agentic_rl_tuning(sample_tuning_model, sample_function_components):
    tuning = AgenticRLTuning()
    tuning.tuning = sample_tuning_model
    tuning.tuning.functions = sample_function_components
    return tuning


class TestAgenticRLTuning:
    def test_initialization(self):
        """Test class initialization"""
        tuning = AgenticRLTuning()
        assert tuning.tuning_id is None
        assert isinstance(tuning.tuning, TuningModel)
        assert tuning.tuning.name == "agentic-rl"

    def test_add_function_components(self):
        """Test adding function components"""
        tuning = AgenticRLTuning()

        # Add Rollout component
        tuning.tuning.add_function_components(
            functype=FunctionType.ROLLOUT,
            classpaths="path/to/rollout.py:RolloutProcessor",
            workspace_dir="./",
        )
        for fc in tuning.tuning.functions:
            fc.fcmodel.filepath, fc.fcmodel.classname = get_filepath_classname(
                fc.fcmodel.classpath,
            )

        assert len(tuning.tuning.functions) == 1
        fc = tuning.tuning.functions[0]
        assert fc.type == FunctionType.ROLLOUT
        assert fc.fcmodel.filepath == "path/to/rollout.py"
        assert fc.fcmodel.classname == "RolloutProcessor"

        # Add multiple Reward components
        tuning.tuning.add_function_components(
            functype=FunctionType.REWARD,
            classpaths=[
                "path/to/reward1.py:Reward1",
                "path/to/reward2.py:Reward2",
            ],
            runtimes=[{"cpu": 1}, {"cpu": 2}],
            workspace_dir="./",
        )
        for fc in tuning.tuning.functions:
            fc.fcmodel.filepath, fc.fcmodel.classname = get_filepath_classname(
                fc.fcmodel.classpath,
            )

        assert len(tuning.tuning.functions) == 3
        assert tuning.tuning.functions[1].type == FunctionType.REWARD
        assert (
            tuning.tuning.functions[1].fcmodel.filepath == "path/to/reward1.py"
        )
        assert tuning.tuning.functions[1].runtime.cpu == 1

        assert tuning.tuning.functions[2].type == FunctionType.REWARD
        assert (
            tuning.tuning.functions[2].fcmodel.filepath == "path/to/reward2.py"
        )
        assert tuning.tuning.functions[2].runtime.cpu == 2

    # pylint: disable=redefined-outer-name
    @pytest.mark.asyncio
    async def test_register_functions_success(self, agentic_rl_tuning):
        """Test successful function component registration"""
        # Create mocks for each component (bypass Pydantic __setattr__)
        for fc in agentic_rl_tuning.tuning.functions:
            object.__setattr__(
                fc,
                "register",
                AsyncMock(
                    return_value=MagicMock(
                        status=MagicMock(success=True),
                        output={"entity_id": f"entity-{fc.type}"},
                    ),
                ),
            )
            object.__setattr__(
                fc,
                "load",
                AsyncMock(
                    return_value=MagicMock(
                        status=MagicMock(success=True),
                        output={"instance_id": f"instance-{fc.type}"},
                    ),
                ),
            )

        # Call registration method
        result = await agentic_rl_tuning.tuning.register_functions(
            lazy_load=False,
        )

        # Verify results
        (
            rollout_ids,
            reward_ids,
            group_reward_ids,
            rollout_instances,
            reward_instances,
            group_reward_instances,
        ) = result

        assert rollout_ids == ["entity-rollout"]
        assert reward_ids == ["entity-reward"]
        assert group_reward_ids == []
        assert rollout_instances == ["instance-rollout"]
        assert reward_instances == ["instance-reward"]
        assert group_reward_instances == []

        # Verify each component called register and load
        for fc in agentic_rl_tuning.tuning.functions:
            fc.register.assert_called_once()
            fc.load.assert_called_once()

    # pylint: disable=redefined-outer-name
    @pytest.mark.asyncio
    async def test_register_functions_failure(self, agentic_rl_tuning):
        """Test function component registration failure"""
        fc0 = agentic_rl_tuning.tuning.functions[0]
        fc1 = agentic_rl_tuning.tuning.functions[1]

        object.__setattr__(
            fc0,
            "register",
            AsyncMock(
                return_value=MagicMock(
                    status=MagicMock(
                        success=False,
                        message="Registration failed",
                    ),
                ),
            ),
        )
        object.__setattr__(
            fc1,
            "register",
            AsyncMock(
                return_value=MagicMock(
                    status=MagicMock(success=True),
                    output={"entity_id": "entity-reward"},
                ),
            ),
        )
        object.__setattr__(
            fc1,
            "load",
            AsyncMock(
                return_value=MagicMock(
                    status=MagicMock(success=False, message="Load failed"),
                ),
            ),
        )

        with pytest.raises(RegistrationError) as exc_info:
            await agentic_rl_tuning.tuning.register_functions()

        assert "Registration failed" in str(exc_info.value)
        assert exc_info.value.error_code == 2052

    # pylint: disable=redefined-outer-name
    @pytest.mark.asyncio
    async def test_register_datasets_success(self, agentic_rl_tuning):
        """Test successful dataset registration"""
        # Patch the to_bailian_data function in the correct location
        with patch(
            "dashscope.finetune.reinforcement.common.model.to_bailian_data",
            new_callable=AsyncMock,
            return_value=["dummy-file-id"],  # Same value for every call
        ):
            (
                train_ids,
                val_ids,
            ) = await agentic_rl_tuning.tuning.upload_datasets()

            assert train_ids == ["dummy-file-id", "dummy-file-id"]
            assert val_ids == ["dummy-file-id"]

            # Verify model state update
            assert (
                agentic_rl_tuning.tuning.datasets[0].file_id == "dummy-file-id"
            )
            assert (
                agentic_rl_tuning.tuning.datasets[0].file_id == "dummy-file-id"
            )

    # pylint: disable=redefined-outer-name
    @pytest.mark.asyncio
    async def test_upload_datasets_failure(self, agentic_rl_tuning):
        """Test dataset registration failure"""
        # Mock file system operations
        with patch("os.path.exists", return_value=True), patch(
            "os.path.isfile",
            return_value=True,
        ), patch("os.path.getsize", return_value=100), patch(
            "dashscope.finetune.reinforcement.common.utils.to_bailian_data",
            new_callable=AsyncMock,
            side_effect=Exception("Upload failed"),
        ):
            with pytest.raises(OSSUploadError) as exc_info:
                await agentic_rl_tuning.tuning.upload_datasets()

            assert "Dataset registration failed" in str(
                exc_info.value,
            )

    # pylint: disable=redefined-outer-name
    def test_get_entity_ids(self, agentic_rl_tuning):
        """Test getting entity IDs"""
        # Set entity IDs
        agentic_rl_tuning.tuning.functions[0].entity_id = "rollout-entity"
        agentic_rl_tuning.tuning.functions[1].entity_id = "reward-entity"

        # Test getting Rollout entity ID
        rollout_ids = agentic_rl_tuning.tuning.get_entity_ids(
            FunctionType.ROLLOUT,
        )
        assert rollout_ids == ["rollout-entity"]

        # Test getting Reward entity ID
        reward_ids = agentic_rl_tuning.tuning.get_entity_ids(
            FunctionType.REWARD,
        )
        assert reward_ids == ["reward-entity"]

        # Test getting Group Reward entity ID (none)
        group_ids = agentic_rl_tuning.tuning.get_entity_ids(
            FunctionType.GROUP_REWARD,
        )
        assert group_ids == []

    # pylint: disable=redefined-outer-name
    def test_get_runtimes(self, agentic_rl_tuning):
        """Test getting runtime configurations"""
        # Test getting Rollout runtime
        rollout_runtimes = agentic_rl_tuning.tuning.get_runtimes(
            FunctionType.ROLLOUT,
        )
        assert len(rollout_runtimes) == 1
        assert rollout_runtimes[0]["cpu"] == 2

        # Test getting Reward runtime
        reward_runtimes = agentic_rl_tuning.tuning.get_runtimes(
            FunctionType.REWARD,
        )
        assert len(reward_runtimes) == 1
        assert reward_runtimes[0]["cpu"] == 1

        # Test getting Group Reward runtime (none)
        group_runtimes = agentic_rl_tuning.tuning.get_runtimes(
            FunctionType.GROUP_REWARD,
        )
        assert group_runtimes == []

    # pylint: disable=redefined-outer-name
    def test_combine_ids_runtimes(self, agentic_rl_tuning):
        """Test combining IDs and runtime configurations"""
        # Set entity IDs
        agentic_rl_tuning.tuning.functions[0].entity_id = "rollout-entity"
        agentic_rl_tuning.tuning.functions[1].entity_id = "reward-entity"

        # Test Rollout combination
        rollout_functions = agentic_rl_tuning.tuning.combine_ids_runtimes(
            FunctionType.ROLLOUT,
            None,  # Use model's IDs
            None,  # Use model's runtime
        )
        assert len(rollout_functions) == 1
        # Check only the fields we care about
        assert rollout_functions[0]["rollout_id"] == "rollout-entity"
        assert rollout_functions[0]["cpu"] == 2
        assert rollout_functions[0]["memory_size"] == 4096
        assert rollout_functions[0]["concurrency"] == 10

        # Test Reward combination (override with parameters)
        reward_functions = agentic_rl_tuning.tuning.combine_ids_runtimes(
            FunctionType.REWARD,
            ["custom-reward-id"],  # Provide custom ID
            [{"cpu": 3}],  # Provide custom runtime
        )
        assert len(reward_functions) == 1
        assert reward_functions[0]["reward_id"] == "custom-reward-id"
        assert reward_functions[0]["cpu"] == 3

        # Test Group Reward combination (no components)
        group_functions = agentic_rl_tuning.tuning.combine_ids_runtimes(
            FunctionType.GROUP_REWARD,
            None,
            None,
        )
        assert group_functions == []

    # pylint: disable=redefined-outer-name
    def test_combine_ids_runtimes_edge_cases(self):
        """Test edge cases for combining IDs and runtime configurations"""
        # Create empty model
        tuning_model = TuningModel()
        # Test different types
        for func_type in FunctionType:
            result = tuning_model.combine_ids_runtimes(func_type, [], [])
            assert not result

        # Test ID and runtime count mismatch
        tuning_model.add_function_components(
            functype=FunctionType.ROLLOUT,
            classpaths=["path1.py:Class1", "path2.py:Class2"],
            runtimes=[{"cpu": 1}],  # Only one runtime provided
            workspace_dir="./",
        )
        assert len(tuning_model.functions) == 2

        # Should have two components but only one runtime
        functions = tuning_model.combine_ids_runtimes(
            FunctionType.ROLLOUT,
            ["rollout-id-1", "rollout-id-2"],
            None,
        )
        assert len(functions) == 2

        # First has runtime
        assert functions[0]["cpu"] == 1

        # Second has no runtime
        assert "cpu" not in functions[1]
