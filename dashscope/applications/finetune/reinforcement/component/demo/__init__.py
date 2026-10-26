# -*- coding: utf-8 -*-
from dashscope.applications.finetune.reinforcement.component.demo.group_reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
    DemoGroupRewardProcessor,
)
from dashscope.applications.finetune.reinforcement.component.demo.reward_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
    DemoRewardProcessor,
)
from dashscope.applications.finetune.reinforcement.component.demo.rollout_processor_demo import (  # noqa: E501  # pylint: disable=line-too-long
    DemoRolloutProcessor,
)

__all__ = [
    "DemoRewardProcessor",
    # Inherits AbstractRewardProcessor, returns RewardOutput
    "DemoRolloutProcessor",
    # Inherits AbstractRolloutProcessor, returns RolloutOutput
    "DemoGroupRewardProcessor",
    # Inherits AbstractGroupRewardProcessor, returns GroupRewardOutput
]
