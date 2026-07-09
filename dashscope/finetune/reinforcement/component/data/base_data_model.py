# -*- coding: utf-8 -*-
"""
component/data/base_data_model.py

Base classes and common components for processor input parameters.
"""

import hashlib
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, SecretStr

from dashscope.finetune.reinforcement.common.model_types import (
    FunctionType as FuncType,
)


# ========================================================================== #
#                              Utility Functions                             #
# ========================================================================== #


def _generate_ro_id(length: int) -> str:
    """Generate rollout ID."""
    return "ro-" + hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:length]


# ========================================================================== #
#                              Enum Definitions                              #
# ========================================================================== #


class TaskStatus(str, Enum):
    """Task execution status."""

    SUCCESS = "success"
    FAILED = "failed"


class ModelProtocol(str, Enum):
    """Model API protocol types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DASHSCOPE = "dashscope"


# ========================================================================== #
#                              Base Class Definitions                        #
# ========================================================================== #


class BaseDataModel(BaseModel):
    """
    Base class for processor input parameters.
    All business-type processor input models should inherit from this class.
    """

    func_type: FuncType

    class Config:
        extra = "allow"


# ========================================================================== #
#                         REQUEST METADATA                                   #
# ========================================================================== #


class ModelResource(BaseModel):
    """
    Defines the configuration resources required for the agent.
    """

    model_name: str = Field(
        ...,
        description="The specific identifier for the model (e.g., 'gpt-4o', "
        "'llama-3-70b').",
    )
    base_url: str = Field(
        ...,
        description="The endpoint URL for the model provider.",
    )
    api_key: SecretStr = Field(
        ...,
        description="The authentication key for the provider.",
    )


class RequestMetadata(BaseModel):
    """
    Defines the metadata required for the request.
    """

    job_id: str = Field(
        ...,
        description="A unique identifier for the job.",
    )
    sample_id: str = Field(
        ...,
        description="A unique identifier for the sample.",
    )
    rollout_id: str = Field(
        ...,
        description="A unique identifier for tracking this specific task "
        "execution.",
    )
    attempt_id: str = Field(
        ...,
        description="A unique identifier for the attempt.",
    )


# ========================================================================== #
#                         AGENT TASK BASE COMPONENTS                         #
# ========================================================================== #


class Task(BaseModel):
    """
    Defines structure for individual training samples or execution tasks.
    """

    rollout_id: str = Field(
        default_factory=lambda: _generate_ro_id(12),
        description="Unique identifier for tracking specific task executions.",
    )

    prompt: Union[str, List[Dict], Dict] = Field(
        ...,
        description="Input instructions/questions/message list provided to "
        "the agent.",
    )

    ground_truth: Optional[Any] = Field(
        default=None,
        description="Reference answer for reward calculation or automated "
        "evaluation.",
    )

    training_state: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional training state information (e.g., task_id, "
        "epoch).",
    )


class Resource(BaseModel):
    """
    Defines configuration workspace required by the agent.
    """

    model_name: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4o', 'qwen-max').",
    )
    base_url: str = Field(
        ...,
        description="Endpoint URL for model service.",
    )
    api_key: SecretStr = Field(
        ...,
        description="Authentication API key for model service.",
    )

    protocol: ModelProtocol = Field(
        default=ModelProtocol.OPENAI,
        description="API protocol standard.",
    )

    max_tokens: int = Field(
        2048,
        ge=1,
        description="Maximum token limit for generated output.",
    )
    max_turns: int = Field(
        25,
        ge=1,
        description="Safety upper limit for dialogue turns.",
    )
    sampling_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sampling parameters (e.g., temperature, top_p).",
    )
    timeout: float = Field(
        60.0,
        description="Request timeout in seconds.",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Global system instruction for the agent.",
    )


class AgentOutput(BaseModel):
    """
    Represents final result of agent execution or specific rollout.
    """

    messages: Optional[List[Dict]] = Field(
        None,
        description="The complete conversation history or sequence of "
        "internal thoughts and actions.",
    )

    rollout_extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra information from the dataset.",
    )

    rollout_metrics: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Additional evaluation metrics or metadata associated "
        "with this output.",
    )

    reward_score: Optional[float] = Field(
        None,
        description="Scalar feedback score assigned to this output ("
        "typically for RL/evaluation).",
    )
