# -*- coding: utf-8 -*-
# ========================================================================== #
#                              Base Components                               #
# ========================================================================== #
from dashscope.applications.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
    TaskStatus,
    ModelProtocol,
    Task,
    Resource,
    ModelResource,
    RequestMetadata,
    AgentOutput,
)
from dashscope.applications.finetune.reinforcement.component.data.group_reward_input import (  # noqa: E501
    GroupRewardInput,
)
from dashscope.applications.finetune.reinforcement.component.data.group_reward_output import (  # noqa: E501
    GroupRewardOutput,
    GroupReward,
)

# ========================================================================== #
#                              Input Models                                  #
# ========================================================================== #
from dashscope.applications.finetune.reinforcement.component.data.reward_input import (
    RewardInput,
)

# ========================================================================== #
#                              Output Models                                 #
# ========================================================================== #
from dashscope.applications.finetune.reinforcement.component.data.reward_output import (
    RewardOutput,
    Reward,
)
from dashscope.applications.finetune.reinforcement.component.data.rollout_input import (
    RolloutInput,
)
from dashscope.applications.finetune.reinforcement.component.data.rollout_output import (
    RolloutOutput,
)

__all__ = [
    # Base Components
    "BaseDataModel",
    "TaskStatus",
    "ModelProtocol",
    "Task",
    "Resource",
    "ModelResource",
    "RequestMetadata",
    "AgentOutput",
    # Input Models
    "RewardInput",
    "RolloutInput",
    "GroupRewardInput",
    # Output Models
    "Reward",
    "RewardOutput",
    "RolloutOutput",
    "GroupReward",
    "GroupRewardOutput",
]
