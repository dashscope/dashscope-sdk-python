# -*- coding: utf-8 -*-
"""
component/data/group_reward_input.py

Data model definitions for GroupReward processor input parameters.
Corresponds to the GroupRewardInput structure in protocol.py.
"""

from typing import Any, List, Optional
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
#                         INPUT: GROUP REWARD REQUEST                        #
# ========================================================================== #


class GroupRewardInput(BaseDataModel):
    """
    Input parameter model for GroupReward business processor.

    Corresponds to the GroupRewardInput structure in protocol.py.
    """

    func_type: FuncType = FuncType.GROUP_REWARD

    agent_outputs: List[AgentOutput] = Field(
        ...,
        description="The agent outputs for the reward calculation.",
    )

    request_metadata: Optional[RequestMetadata] = Field(
        default=None,
        description="The metadata for the request.",
    )

    ground_truth: Optional[Any] = Field(
        default=None,
        description="The expected gold-standard answer used for reward "
        "calculation or automated evaluation.",
    )

    model_config = ConfigDict(extra="allow")
