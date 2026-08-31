# -*- coding: utf-8 -*-
"""
component/data/reward_output.py

Data model definitions for Reward processor output results.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field

from dashscope.finetune.reinforcement.component.data.base_data_model import (
    TaskStatus,
)


# ========================================================================== #
#                              REWARD COMPONENTS                             #
# ========================================================================== #


class Reward(BaseModel):
    """Reward calculation result."""

    reward_score: float = Field(
        ...,
        description="The reward score.",
    )

    reward_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Additional reward-specific metrics as string key-value "
        "pairs.",
    )


# ========================================================================== #
#                              OUTPUT: REWARD RESPONSE                       #
# ========================================================================== #


class RewardOutput(BaseModel):
    """
    Reward processor output result model.
    """

    reward: Reward = Field(
        ...,
        description="The computed reward for the given rollout.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.SUCCESS,
        description="The status of the reward computation.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if the reward computation failed.",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code if the reward computation failed.",
    )

    class Config:
        extra = "allow"
