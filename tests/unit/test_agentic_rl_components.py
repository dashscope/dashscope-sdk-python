# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
import copy
from typing import Dict

import pytest
from pydantic import SecretStr, ValidationError

from dashscope.finetune.reinforcement import (
    AbstractGroupRewardProcessor,
    AbstractRewardProcessor,
    AbstractRolloutProcessor,
    AgentOutput,
    GroupReward,
    GroupRewardInput,
    GroupRewardOutput,
    Reward,
    RewardInput,
    RewardOutput,
    RolloutInput,
    RolloutOutput,
    TaskStatus,
    aggregate_func,
    reward_func,
    sub_reward_func,
)
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    ModelProtocol,
    ModelResource,
    RequestMetadata,
    Resource,
    Task,
)
from dashscope.finetune.reinforcement.component.func_decorator import (
    AggregateFunction,
    RewardProcessorMeta,
    SubRewardFunction,
)
from dashscope.finetune.reinforcement.component.parser.group_reward_parser import (  # noqa: E501  # pylint: disable=line-too-long
    GroupRewardRequestParser,
)
from dashscope.finetune.reinforcement.component.parser.reward_parser import (  # noqa: E501  # pylint: disable=line-too-long
    RewardRequestParser,
)
from dashscope.finetune.reinforcement.component.parser.rollout_parser import (  # noqa: E501  # pylint: disable=line-too-long
    RolloutRequestParser,
)


# ========================================================================== #
#                              Fixtures                                      #
# ========================================================================== #


@pytest.fixture
def model_resource():
    return ModelResource(
        model_name="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-test-key"),
    )


@pytest.fixture
def request_metadata():
    return RequestMetadata(
        job_id="ft-12345",
        sample_id="sample-001",
        rollout_id="ro-abc",
        attempt_id="att-001",
    )


@pytest.fixture
def agent_output():
    return AgentOutput(
        messages=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
        reward_score=None,
        rollout_metrics={"latency": 0.5},
    )


@pytest.fixture
def rollout_input(model_resource, request_metadata):
    return RolloutInput(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model_resource=model_resource,
        request_metadata=request_metadata,
        ground_truth="4",
    )


@pytest.fixture
def reward_input(agent_output, request_metadata):
    return RewardInput(
        agent_output=agent_output,
        ground_truth="4",
        request_metadata=request_metadata,
    )


@pytest.fixture
def group_reward_input(request_metadata):
    outputs = [
        AgentOutput(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            reward_score=None,
        ),
        AgentOutput(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "I don't know"},
            ],
            reward_score=None,
        ),
    ]
    return GroupRewardInput(
        agent_outputs=outputs,
        ground_truth="4",
        request_metadata=request_metadata,
    )


# ========================================================================== #
#                         Data Model Tests                                    #
# ========================================================================== #


class TestBaseDataModels:
    def test_task_status_values(self):
        assert TaskStatus.SUCCESS == "success"
        assert TaskStatus.FAILED == "failed"

    def test_model_protocol_values(self):
        assert ModelProtocol.OPENAI == "openai"
        assert ModelProtocol.ANTHROPIC == "anthropic"
        assert ModelProtocol.DASHSCOPE == "dashscope"

    def test_model_resource_creation(self, model_resource):
        assert model_resource.model_name == "qwen-max"
        assert model_resource.api_key.get_secret_value() == "sk-test-key"

    def test_request_metadata_creation(self, request_metadata):
        assert request_metadata.job_id == "ft-12345"
        assert request_metadata.sample_id == "sample-001"
        assert request_metadata.rollout_id == "ro-abc"
        assert request_metadata.attempt_id == "att-001"

    def test_task_auto_generates_rollout_id(self):
        task = Task(prompt="What is 2+2?")
        assert task.rollout_id.startswith("ro-")
        assert len(task.rollout_id) > 3

    def test_task_with_all_fields(self):
        task = Task(
            prompt=[{"role": "user", "content": "Hello"}],
            ground_truth="Hi",
            training_state={"epoch": 1},
        )
        assert task.ground_truth == "Hi"
        assert task.training_state == {"epoch": 1}

    def test_task_unique_rollout_ids(self):
        task1 = Task(prompt="a")
        task2 = Task(prompt="b")
        assert task1.rollout_id != task2.rollout_id

    def test_resource_defaults(self):
        resource = Resource(
            model_name="qwen-max",
            base_url="https://example.com",
            api_key=SecretStr("key"),
        )
        assert resource.protocol == ModelProtocol.OPENAI
        assert resource.max_tokens == 2048
        assert resource.max_turns == 25
        assert resource.timeout == 60.0
        assert resource.system_prompt is None
        assert resource.sampling_params == {}

    def test_resource_custom_values(self):
        resource = Resource(
            model_name="qwen-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=SecretStr("sk-key"),
            protocol=ModelProtocol.DASHSCOPE,
            max_tokens=4096,
            max_turns=50,
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            timeout=120.0,
            system_prompt="You are a helpful assistant.",
        )
        assert resource.max_tokens == 4096
        assert resource.sampling_params["temperature"] == 0.7
        assert resource.system_prompt == "You are a helpful assistant."

    def test_agent_output_creation(self, agent_output):
        assert len(agent_output.messages) == 2
        assert agent_output.reward_score is None
        assert agent_output.rollout_metrics["latency"] == 0.5

    def test_agent_output_defaults(self):
        output = AgentOutput()
        assert output.messages is None
        assert output.reward_score is None
        assert output.rollout_metrics == {}
        assert output.rollout_extra is None


class TestRolloutDataModels:
    def test_rollout_input_creation(self, rollout_input):
        assert rollout_input.func_type.value == "rollout"
        assert len(rollout_input.messages) == 1
        assert rollout_input.ground_truth == "4"
        assert rollout_input.model_resource.model_name == "qwen-max"

    def test_rollout_input_optional_fields(self, model_resource):
        ri = RolloutInput(
            messages=[{"role": "user", "content": "Hello"}],
            model_resource=model_resource,
        )
        assert ri.tools is None
        assert ri.ground_truth is None
        assert ri.rollout_extra is None
        assert ri.sampling_params is None
        assert ri.request_metadata is None

    def test_rollout_input_with_tools(self, model_resource):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object"},
                },
            },
        ]
        ri = RolloutInput(
            messages=[{"role": "user", "content": "Calculate 2+2"}],
            model_resource=model_resource,
            tools=tools,
        )
        assert len(ri.tools) == 1
        assert ri.tools[0]["function"]["name"] == "calculator"

    def test_rollout_input_extra_fields_allowed(self, model_resource):
        ri = RolloutInput(
            messages=[{"role": "user", "content": "Hello"}],
            model_resource=model_resource,
            custom_field="extra_data",
        )
        assert ri.custom_field == "extra_data"

    def test_rollout_input_missing_required_fields(self):
        with pytest.raises(ValidationError):
            RolloutInput(messages=[{"role": "user", "content": "Hello"}])

    def test_rollout_output_success(self, agent_output):
        output = RolloutOutput(
            agent_output=agent_output,
            status=TaskStatus.SUCCESS,
        )
        assert output.status == TaskStatus.SUCCESS
        assert output.error is None
        assert output.agent_output.messages is not None

    def test_rollout_output_failure(self):
        output = RolloutOutput(
            status=TaskStatus.FAILED,
            error="Model inference timeout",
        )
        assert output.status == TaskStatus.FAILED
        assert output.error == "Model inference timeout"
        assert output.agent_output is None

    def test_rollout_output_defaults(self):
        output = RolloutOutput()
        assert output.status == TaskStatus.SUCCESS
        assert output.agent_output is None
        assert output.error is None


class TestRewardDataModels:
    def test_reward_creation(self):
        reward = Reward(reward_score=0.85)
        assert reward.reward_score == 0.85
        assert reward.reward_metrics is None

    def test_reward_with_metrics(self):
        reward = Reward(
            reward_score=0.9,
            reward_metrics={"accuracy": 0.95, "fluency": 0.85},
        )
        assert reward.reward_metrics["accuracy"] == 0.95

    def test_reward_input_creation(self, reward_input):
        assert reward_input.func_type.value == "reward"
        assert reward_input.ground_truth == "4"
        assert reward_input.agent_output.messages is not None

    def test_reward_input_without_ground_truth(self, agent_output):
        ri = RewardInput(agent_output=agent_output)
        assert ri.ground_truth is None

    def test_reward_input_missing_agent_output(self):
        with pytest.raises(ValidationError):
            RewardInput()

    def test_reward_input_extra_fields_allowed(self, agent_output):
        ri = RewardInput(
            agent_output=agent_output,
            custom_metric="test",
        )
        assert ri.custom_metric == "test"

    def test_reward_output_success(self):
        output = RewardOutput(
            reward=Reward(reward_score=1.0),
            status=TaskStatus.SUCCESS,
        )
        assert output.reward.reward_score == 1.0
        assert output.status == TaskStatus.SUCCESS

    def test_reward_output_failure(self):
        output = RewardOutput(
            reward=Reward(reward_score=0.0),
            status=TaskStatus.FAILED,
            error="Computation error",
        )
        assert output.status == TaskStatus.FAILED

    def test_reward_output_missing_reward(self):
        with pytest.raises(ValidationError):
            RewardOutput()


class TestGroupRewardDataModels:
    def test_group_reward_input_creation(self, group_reward_input):
        assert group_reward_input.func_type.value == "group_reward"
        assert len(group_reward_input.agent_outputs) == 2
        assert group_reward_input.ground_truth == "4"

    def test_group_reward_input_missing_outputs(self):
        with pytest.raises(ValidationError):
            GroupRewardInput()

    def test_group_reward_output_success(self):
        output = GroupRewardOutput(
            reward=GroupReward(
                rewards=[
                    Reward(reward_score=1.0),
                    Reward(reward_score=0.5),
                ],
            ),
            status=TaskStatus.SUCCESS,
        )
        assert len(output.reward.rewards) == 2
        assert output.reward.rewards[0].reward_score == 1.0
        assert output.reward.rewards[1].reward_score == 0.5

    def test_group_reward_output_empty_rewards(self):
        output = GroupRewardOutput(
            reward=GroupReward(rewards=[]),
            status=TaskStatus.SUCCESS,
        )
        assert len(output.reward.rewards) == 0

    def test_group_reward_output_failure(self):
        output = GroupRewardOutput(
            reward=GroupReward(rewards=[]),
            status=TaskStatus.FAILED,
            error="Group computation failed",
        )
        assert output.status == TaskStatus.FAILED


# ========================================================================== #
#                         Parser Tests                                        #
# ========================================================================== #


class TestParsers:
    def test_rollout_parser(self):
        parser = RolloutRequestParser()
        raw = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model_resource": {
                "model_name": "qwen-max",
                "base_url": "https://example.com",
                "api_key": "sk-test",
            },
            "ground_truth": "Hi",
        }
        result = parser.parse(raw)
        assert isinstance(result, RolloutInput)
        assert result.messages[0]["content"] == "Hello"
        assert result.model_resource.model_name == "qwen-max"

    def test_rollout_parser_missing_required(self):
        parser = RolloutRequestParser()
        with pytest.raises(Exception):
            parser.parse({"messages": []})

    def test_reward_parser(self):
        parser = RewardRequestParser()
        raw = {
            "agent_output": {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "reward_score": None,
            },
            "ground_truth": "Answer",
        }
        result = parser.parse(raw)
        assert isinstance(result, RewardInput)
        assert result.ground_truth == "Answer"
        assert len(result.agent_output.messages) == 2

    def test_reward_parser_missing_agent_output(self):
        parser = RewardRequestParser()
        with pytest.raises(Exception):
            parser.parse({"ground_truth": "test"})

    def test_group_reward_parser(self):
        parser = GroupRewardRequestParser()
        raw = {
            "agent_outputs": [
                {
                    "messages": [{"role": "assistant", "content": "A"}],
                    "reward_score": None,
                },
                {
                    "messages": [{"role": "assistant", "content": "B"}],
                    "reward_score": None,
                },
            ],
            "ground_truth": "A",
        }
        result = parser.parse(raw)
        assert isinstance(result, GroupRewardInput)
        assert len(result.agent_outputs) == 2

    def test_group_reward_parser_missing_outputs(self):
        parser = GroupRewardRequestParser()
        with pytest.raises(Exception):
            parser.parse({"ground_truth": "test"})

    def test_rollout_parser_extra_fields(self):
        parser = RolloutRequestParser()
        raw = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model_resource": {
                "model_name": "qwen-max",
                "base_url": "https://example.com",
                "api_key": "sk-test",
            },
            "custom_field": "extra",
        }
        result = parser.parse(raw)
        assert result.custom_field == "extra"


# ========================================================================== #
#                     Processor Tests (Demo Implementations)                 #
# ========================================================================== #


class TestDemoRolloutProcessor:
    def test_setup(self):
        from dashscope.finetune.reinforcement.component.demo.rollout_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRolloutProcessor,
        )

        processor = DemoRolloutProcessor()
        processor.setup()

    def test_process(self, rollout_input):
        from dashscope.finetune.reinforcement.component.demo.rollout_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRolloutProcessor,
        )

        processor = DemoRolloutProcessor()
        processor.setup()
        result = processor.process(rollout_input)
        assert isinstance(result, RolloutOutput)
        assert result.status == TaskStatus.SUCCESS
        assert result.error is None
        assert result.agent_output is not None
        assert any(
            msg.get("role") == "assistant"
            for msg in result.agent_output.messages
        )

    def test_process_echoes_last_user_message(self, model_resource):
        from dashscope.finetune.reinforcement.component.demo.rollout_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRolloutProcessor,
        )

        processor = DemoRolloutProcessor()
        ri = RolloutInput(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello world!"},
            ],
            model_resource=model_resource,
        )
        result = processor.process(ri)
        assistant_msgs = [
            m for m in result.agent_output.messages if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert "Hello world!" in assistant_msgs[0]["content"]


class TestDemoRewardProcessor:
    def test_setup(self):
        from dashscope.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRewardProcessor,
        )

        processor = DemoRewardProcessor()
        processor.setup()

    def test_process_ground_truth_match(self):
        from dashscope.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRewardProcessor,
        )

        processor = DemoRewardProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4."},
                ],
            ),
            ground_truth="4",
        )
        result = processor.process(ri)
        assert isinstance(result, RewardOutput)
        assert result.status == TaskStatus.SUCCESS
        assert result.reward.reward_score == 1.0

    def test_process_ground_truth_no_match(self):
        from dashscope.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRewardProcessor,
        )

        processor = DemoRewardProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "I don't know."},
                ],
            ),
            ground_truth="4",
        )
        result = processor.process(ri)
        assert result.reward.reward_score == 0.5

    def test_process_no_ground_truth(self):
        from dashscope.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRewardProcessor,
        )

        processor = DemoRewardProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[
                    {"role": "assistant", "content": "Hello"},
                ],
            ),
        )
        result = processor.process(ri)
        assert result.reward.reward_score == 0.0

    def test_process_empty_messages(self):
        from dashscope.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoRewardProcessor,
        )

        processor = DemoRewardProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(messages=[]),
            ground_truth="4",
        )
        result = processor.process(ri)
        assert result.reward.reward_score == 0.0


class TestDemoGroupRewardProcessor:
    def test_setup(self):
        from dashscope.finetune.reinforcement.component.demo.group_reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoGroupRewardProcessor,
        )

        processor = DemoGroupRewardProcessor()
        processor.setup()

    def test_process_mixed_results(self, group_reward_input):
        from dashscope.finetune.reinforcement.component.demo.group_reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoGroupRewardProcessor,
        )

        processor = DemoGroupRewardProcessor()
        result = processor.process(group_reward_input)
        assert isinstance(result, GroupRewardOutput)
        assert result.status == TaskStatus.SUCCESS
        assert len(result.reward.rewards) == 2
        assert result.reward.rewards[0].reward_score == 1.0
        assert result.reward.rewards[1].reward_score == 0.5

    def test_process_empty_outputs(self, request_metadata):
        from dashscope.finetune.reinforcement.component.demo.group_reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoGroupRewardProcessor,
        )

        processor = DemoGroupRewardProcessor()
        gi = GroupRewardInput(
            agent_outputs=[],
            request_metadata=request_metadata,
        )
        result = processor.process(gi)
        assert result.status == TaskStatus.SUCCESS
        assert len(result.reward.rewards) == 0

    def test_process_all_correct(self, request_metadata):
        from dashscope.finetune.reinforcement.component.demo.group_reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
            DemoGroupRewardProcessor,
        )

        processor = DemoGroupRewardProcessor()
        gi = GroupRewardInput(
            agent_outputs=[
                AgentOutput(
                    messages=[{"role": "assistant", "content": "answer is 4"}],
                ),
                AgentOutput(
                    messages=[{"role": "assistant", "content": "result: 4"}],
                ),
            ],
            ground_truth="4",
            request_metadata=request_metadata,
        )
        result = processor.process(gi)
        assert all(r.reward_score == 1.0 for r in result.reward.rewards)


# ========================================================================== #
#                    Custom Processor Implementation Tests                   #
# ========================================================================== #


class TestCustomRolloutProcessor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            # pylint: disable=abstract-class-instantiated
            AbstractRolloutProcessor()

    @pytest.mark.asyncio
    async def test_custom_rollout_processor(self, rollout_input):
        class MyRolloutProcessor(AbstractRolloutProcessor):
            async def process(self, input_data: RolloutInput) -> RolloutOutput:
                messages = list(input_data.messages)
                messages.append(
                    {"role": "assistant", "content": "custom response"},
                )
                return RolloutOutput(
                    agent_output=AgentOutput(messages=messages),
                    status=TaskStatus.SUCCESS,
                )

        processor = MyRolloutProcessor()
        result = await processor.process(rollout_input)
        assert result.status == TaskStatus.SUCCESS
        assert result.agent_output.messages[-1]["content"] == "custom response"

    @pytest.mark.asyncio
    async def test_custom_rollout_with_setup(self, rollout_input):
        class StatefulRolloutProcessor(AbstractRolloutProcessor):
            def setup(self) -> None:
                self.initialized = True

            async def process(self, input_data: RolloutInput) -> RolloutOutput:
                assert self.initialized
                return RolloutOutput(
                    agent_output=AgentOutput(messages=input_data.messages),
                    status=TaskStatus.SUCCESS,
                )

        processor = StatefulRolloutProcessor()
        processor.setup()
        result = await processor.process(rollout_input)
        assert result.status == TaskStatus.SUCCESS


class TestCustomRewardProcessor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            # pylint: disable=abstract-class-instantiated
            AbstractRewardProcessor()

    def test_custom_reward_processor(self, reward_input):
        class MyRewardProcessor(AbstractRewardProcessor):
            def process(self, input_data: RewardInput) -> RewardOutput:
                score = 1.0 if input_data.ground_truth is not None else 0.0
                return RewardOutput(
                    reward=Reward(reward_score=score),
                    status=TaskStatus.SUCCESS,
                )

        processor = MyRewardProcessor()
        result = processor.process(reward_input)
        assert result.reward.reward_score == 1.0

    def test_reward_processor_shutdown(self):
        class SimpleReward(AbstractRewardProcessor):
            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = SimpleReward()
        assert processor.executor is not None
        processor.shutdown()
        assert processor.executor is None


class TestCustomGroupRewardProcessor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            # pylint: disable=abstract-class-instantiated
            AbstractGroupRewardProcessor()

    @pytest.mark.asyncio
    async def test_custom_group_reward_processor(self, group_reward_input):
        class MyGroupRewardProcessor(AbstractGroupRewardProcessor):
            async def process(
                self,
                input_data: GroupRewardInput,
            ) -> GroupRewardOutput:
                rewards = [
                    Reward(reward_score=float(i))
                    for i in range(len(input_data.agent_outputs))
                ]
                return GroupRewardOutput(
                    reward=GroupReward(rewards=rewards),
                    status=TaskStatus.SUCCESS,
                )

        processor = MyGroupRewardProcessor()
        result = await processor.process(group_reward_input)
        assert len(result.reward.rewards) == 2
        assert result.reward.rewards[0].reward_score == 0.0
        assert result.reward.rewards[1].reward_score == 1.0


# ========================================================================== #
#                         Decorator Tests                                     #
# ========================================================================== #


class TestSubRewardFunction:
    def test_creation(self):
        def dummy_func():
            return RewardOutput(
                reward=Reward(reward_score=1.0),
                status=TaskStatus.SUCCESS,
            )

        sub = SubRewardFunction(name="test", func=dummy_func, weight=0.5)
        assert sub.name == "test"
        assert sub.weight == 0.5
        assert sub.score == 0.0

    def test_deepcopy_resets_state(self):
        def dummy_func():
            pass

        sub = SubRewardFunction(name="test", func=dummy_func, weight=0.7)
        sub.score = 5.0
        sub.reward_metrics = {"accuracy": 0.9}

        copied = copy.deepcopy(sub)
        assert copied.name == "test"
        assert copied.weight == 0.7
        assert copied.score == 0.0
        assert copied.reward_metrics == {}
        assert copied.func is sub.func


class TestAggregateFunction:
    def test_creation(self):
        def agg_func():
            pass

        agg = AggregateFunction(func=agg_func)
        assert agg.func is agg_func

    def test_deepcopy(self):
        def agg_func():
            pass

        agg = AggregateFunction(func=agg_func)
        copied = copy.deepcopy(agg)
        assert copied.func is agg.func


class TestRewardProcessorMeta:
    def test_creation(self):
        meta = RewardProcessorMeta(processor_id="test-processor")
        assert meta.processor_id == "test-processor"
        assert meta.sub_functions == {}
        assert meta.aggregate_function is None

    def test_deepcopy(self):
        meta = RewardProcessorMeta(processor_id="test")
        meta.sub_functions["sub1"] = SubRewardFunction(
            name="sub1",
            func=lambda x: x,
            weight=0.5,
        )
        copied = copy.deepcopy(meta)
        assert copied.processor_id == "test"
        assert "sub1" in copied.sub_functions
        assert copied.sub_functions["sub1"].weight == 0.5

    def test_copy_caching(self):
        meta = RewardProcessorMeta(processor_id="test")
        copy1 = meta.copy()
        copy2 = meta.copy()
        assert copy1 is copy2


class TestRewardFuncDecorator:
    def test_basic_decorator(self):
        @reward_func("safety")
        class SafetyProcessor(AbstractRewardProcessor):
            @sub_reward_func("toxicity", sub_weight=0.7)
            def toxicity(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.8),
                    status=TaskStatus.SUCCESS,
                )

            @sub_reward_func("refusal", sub_weight=0.3)
            def refusal(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.6),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = SafetyProcessor()
        assert hasattr(processor, "reward_meta")
        assert processor.reward_meta.processor_id == "safety"

    @pytest.mark.asyncio
    async def test_decorator_process_weighted_sum(self):
        @reward_func("quality")
        class QualityProcessor(AbstractRewardProcessor):
            # pylint: disable=unused-argument
            @sub_reward_func("accuracy", sub_weight=0.6)
            def accuracy(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=1.0),
                    status=TaskStatus.SUCCESS,
                )

            # pylint: disable=unused-argument
            @sub_reward_func("fluency", sub_weight=0.4)
            def fluency(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.5),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = QualityProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[{"role": "assistant", "content": "test"}],
            ),
        )
        result = await processor.process(ri)
        assert result.status == TaskStatus.SUCCESS
        expected = 1.0 * 0.6 + 0.5 * 0.4
        assert abs(result.reward.reward_score - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_decorator_with_async_sub_funcs(self):
        @reward_func("async-test")
        class AsyncProcessor(AbstractRewardProcessor):
            # pylint: disable=unused-argument
            @sub_reward_func("sub1", sub_weight=1.0)
            async def sub1(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.9),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = AsyncProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[{"role": "assistant", "content": "test"}],
            ),
        )
        result = await processor.process(ri)
        assert abs(result.reward.reward_score - 0.9) < 1e-6

    @pytest.mark.asyncio
    async def test_decorator_with_custom_aggregate(self):
        @reward_func("custom-agg")
        class CustomAggProcessor(AbstractRewardProcessor):
            # pylint: disable=unused-argument
            @sub_reward_func("sub1", sub_weight=0.5)
            def sub1(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.8),
                    status=TaskStatus.SUCCESS,
                )

            # pylint: disable=unused-argument
            @sub_reward_func("sub2", sub_weight=0.5)
            def sub2(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.6),
                    status=TaskStatus.SUCCESS,
                )

            @aggregate_func
            def aggregate(
                self,
                sub_rewards: Dict[str, RewardOutput],
            ) -> RewardOutput:
                scores = self.get_scores(sub_rewards)
                max_score = max(scores.values())
                return RewardOutput(
                    reward=Reward(reward_score=max_score),
                    status=TaskStatus.SUCCESS,
                )

        # pylint: disable=abstract-class-instantiated
        processor = CustomAggProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[{"role": "assistant", "content": "test"}],
            ),
        )
        result = await processor.process(ri)
        assert result.reward.reward_score == 0.8

    @pytest.mark.asyncio
    async def test_decorator_sub_func_error_handling(self):
        @reward_func("error-test")
        class ErrorProcessor(AbstractRewardProcessor):
            # pylint: disable=unused-argument
            @sub_reward_func("failing", sub_weight=0.5)
            def failing(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                raise RuntimeError("Computation failed")

            # pylint: disable=unused-argument
            @sub_reward_func("working", sub_weight=0.5)
            def working(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=1.0),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = ErrorProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[{"role": "assistant", "content": "test"}],
            ),
        )
        result = await processor.process(ri)
        assert result.status == TaskStatus.SUCCESS
        expected = 0.0 * 0.5 + 1.0 * 0.5
        assert abs(result.reward.reward_score - expected) < 1e-6

    def test_sub_reward_func_default_name(self):
        @reward_func("default-name")
        class DefaultNameProcessor(AbstractRewardProcessor):
            @sub_reward_func(sub_weight=1.0)
            def my_metric(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.5),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = DefaultNameProcessor()
        assert "my_metric" in processor.reward_meta.sub_functions

    def test_get_weights(self):
        @reward_func("weights-test")
        class WeightsProcessor(AbstractRewardProcessor):
            @sub_reward_func("a", sub_weight=0.3)
            def metric_a(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            @sub_reward_func("b", sub_weight=0.7)
            def metric_b(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = WeightsProcessor()
        weights = processor.get_weights()
        assert weights["a"] == 0.3
        assert weights["b"] == 0.7

    def test_get_scores(self):
        @reward_func("scores-test")
        class ScoresProcessor(AbstractRewardProcessor):
            @sub_reward_func("a", sub_weight=0.5)
            def metric_a(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = ScoresProcessor()
        sub_rewards = {
            "a": RewardOutput(
                reward=Reward(reward_score=0.9),
                status=TaskStatus.SUCCESS,
            ),
        }
        scores = processor.get_scores(sub_rewards)
        assert scores["a"] == 0.9

    def test_get_total(self):
        @reward_func("total-test")
        class TotalProcessor(AbstractRewardProcessor):
            @sub_reward_func("a", sub_weight=0.6)
            def metric_a(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            @sub_reward_func("b", sub_weight=0.4)
            def metric_b(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = TotalProcessor()
        sub_rewards = {
            "a": RewardOutput(
                reward=Reward(reward_score=1.0),
                status=TaskStatus.SUCCESS,
            ),
            "b": RewardOutput(
                reward=Reward(reward_score=0.5),
                status=TaskStatus.SUCCESS,
            ),
        }
        total = processor.get_total(sub_rewards)
        expected = 1.0 * 0.6 + 0.5 * 0.4
        assert abs(total - expected) < 1e-6

    def test_get_total_unknown_sub_reward(self):
        @reward_func("unknown-test")
        class UnknownProcessor(AbstractRewardProcessor):
            @sub_reward_func("a", sub_weight=1.0)
            def metric_a(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = UnknownProcessor()
        sub_rewards = {
            "unknown": RewardOutput(
                reward=Reward(reward_score=1.0),
                status=TaskStatus.SUCCESS,
            ),
        }
        with pytest.raises(ValueError, match="not registered"):
            processor.get_total(sub_rewards)

    def test_get_reward_metrics(self):
        @reward_func("metrics-test")
        class MetricsProcessor(AbstractRewardProcessor):
            @sub_reward_func("a", sub_weight=1.0)
            def metric_a(self) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = MetricsProcessor()
        sub_rewards = {
            "a": RewardOutput(
                reward=Reward(
                    reward_score=0.9,
                    reward_metrics={"precision": 0.95, "recall": 0.85},
                ),
                status=TaskStatus.SUCCESS,
            ),
        }
        metrics = processor.get_reward_metrics(sub_rewards)
        assert metrics["a.precision"] == 0.95
        assert metrics["a.recall"] == 0.85

    @pytest.mark.asyncio
    async def test_decorator_with_async_aggregate(self):
        @reward_func("async-agg")
        class AsyncAggProcessor(AbstractRewardProcessor):
            # pylint: disable=unused-argument
            @sub_reward_func("sub1", sub_weight=1.0)
            def sub1(
                self,
                input_data: RewardInput,
            ) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.7),
                    status=TaskStatus.SUCCESS,
                )

            @aggregate_func
            async def aggregate(
                self,
                sub_rewards: Dict[str, RewardOutput],
            ) -> RewardOutput:
                total = sum(
                    r.reward.reward_score for r in sub_rewards.values()
                )
                return RewardOutput(
                    reward=Reward(reward_score=total * 2),
                    status=TaskStatus.SUCCESS,
                )

            def process(self, input_data: RewardInput) -> RewardOutput:
                return RewardOutput(
                    reward=Reward(reward_score=0.0),
                    status=TaskStatus.SUCCESS,
                )

        processor = AsyncAggProcessor()
        ri = RewardInput(
            agent_output=AgentOutput(
                messages=[{"role": "assistant", "content": "test"}],
            ),
        )
        result = await processor.process(ri)
        assert abs(result.reward.reward_score - 1.4) < 1e-6
