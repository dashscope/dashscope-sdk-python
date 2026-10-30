# -*- coding: utf-8 -*-
from .group_reward_processor_demo import DemoGroupRewardProcessor
from .reward_processor_demo import DemoRewardProcessor
from .rollout_processor_demo import DemoRolloutProcessor

__all__ = [
    "DemoRewardProcessor",
    # Inherits AbstractRewardProcessor, returns RewardOutput
    "DemoRolloutProcessor",
    # Inherits AbstractRolloutProcessor, returns RolloutOutput
    "DemoGroupRewardProcessor",
    # Inherits AbstractGroupRewardProcessor, returns GroupRewardOutput
]
