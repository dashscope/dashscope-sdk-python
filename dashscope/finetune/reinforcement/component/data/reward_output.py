# -*- coding: utf-8 -*-
"""
component/data/reward_output.py

Data model definitions for Reward processor output results.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator

from dashscope.finetune.reinforcement.component.data.base_data_model import (
    TaskStatus,
    _normalize_error_code,
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

    @field_validator("error_code", mode="before")
    @classmethod
    def normalize_error_code(cls, v: Any) -> Any:
        """Normalize int error codes to str (protocol stays string)."""
        return _normalize_error_code(v)

    class Config:
        extra = "allow"
