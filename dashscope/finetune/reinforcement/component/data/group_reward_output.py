# -*- coding: utf-8 -*-
"""
component/data/group_reward_output.py

Data model definitions for GroupReward processor output results.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from dashscope.finetune.reinforcement.component.data.base_data_model import (
    TaskStatus,
)
from dashscope.finetune.reinforcement.component.data.reward_output import (
    Reward,
)


# ========================================================================== #
#                         GROUP REWARD COMPONENTS                            #
# ========================================================================== #


class GroupReward(BaseModel):
    """Group reward calculation result."""

    rewards: List[Reward] = Field(
        ...,
        description="The computed rewards for the given rollout.",
    )


# ========================================================================== #
#                       OUTPUT: GROUP REWARD RESPONSE                        #
# ========================================================================== #


class GroupRewardOutput(BaseModel):
    """
    GroupReward processor output result model.
    """

    reward: GroupReward = Field(
        ...,
        description="The computed group reward for the given rollout.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.SUCCESS,
        description="The status of the group reward computation.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if the reward computation failed.",
    )

    model_config = ConfigDict(extra="allow")
