# -*- coding: utf-8 -*-
"""
component/data/reward_input.py

Data model definitions for Reward processor input parameters.
Corresponds to the RewardInput structure in protocol.py.
"""

from typing import Any, Optional
from pydantic import ConfigDict, Field

from dashscope.finetune.reinforcement.common.model_types import (
    FunctionType as FuncType,
)
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
    AgentOutput,
    RequestMetadata,
)


# ========================================================================== #
#                              INPUT: REWARD REQUEST                         #
# ========================================================================== #


class RewardInput(BaseDataModel):
    """
    Input parameter model for Reward business processor.

    Corresponds to the RewardInput structure in protocol.py.
    """

    func_type: FuncType = FuncType.REWARD

    ground_truth: Optional[Any] = Field(
        default=None,
        description="The expected gold-standard answer used for reward "
        "calculation or automated evaluation.",
    )

    request_metadata: Optional[RequestMetadata] = Field(
        default=None,
        description="The metadata for the request.",
    )

    agent_output: AgentOutput = Field(
        ...,
        description="The agent output for the reward calculation.",
    )

    model_config = ConfigDict(extra="allow")
