# -*- coding: utf-8 -*-
"""
component/data/rollout_output.py

Data model definitions for Rollout processor output results.
Corresponds to the RolloutOutput structure in protocol.py.
"""

from typing import Optional
from pydantic import BaseModel, Field

from dashscope.finetune.reinforcement.component.data.base_data_model import (
    TaskStatus,
    AgentOutput,
)


# ========================================================================== #
#                              OUTPUT: ROLLOUT RESULT                        #
# ========================================================================== #


class RolloutOutput(BaseModel):
    """
    Rollout processor output result model.

    Result of agent invocation, returned by AgentClient (remote/local).
    Corresponds to the RolloutOutput structure in protocol.py.
    """

    agent_output: Optional[AgentOutput] = Field(
        default=None,
        description="Agent output content.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.SUCCESS,
        description="Task execution status.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details (when failed).",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code (when failed).",
    )

    class Config:
        extra = "allow"
