# -*- coding: utf-8 -*-
"""
component/data/rollout_input.py

Data model definitions for Rollout processor input parameters.
Corresponds to the RolloutInput structure in protocol.py.
"""

from typing import Any, Dict, List, Optional
from pydantic import ConfigDict, Field

from dashscope.finetune.reinforcement.common.model_types import (
    FunctionType as FuncType,
)
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
    ModelResource,
    RequestMetadata,
)


# ========================================================================== #
#                              INPUT: ROLLOUT JOB                            #
# ========================================================================== #


class RolloutInput(BaseDataModel):
    """
    Input parameter model for Rollout business processor.

    Defines the structure of a single training sample or execution task.
    Corresponds to the RolloutInput structure in protocol.py.
    """

    func_type: FuncType = FuncType.ROLLOUT

    # Dataset related
    messages: List[Dict] = Field(
        ...,
        description="The input messages provided to the agent.",
    )

    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="The tool used by the agent.",
    )

    ground_truth: Optional[Any] = Field(
        default=None,
        description="The expected gold-standard answer used for reward "
        "calculation or automated evaluation.",
    )

    # extra: Extra information from dataset, passed through to rollout and
    # reward
    rollout_extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra information from the dataset.",
    )

    # Inference hyper_parameters
    sampling_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sampling parameters for the agent.",
    )

    model_resource: ModelResource = Field(
        ...,
        description="The model resource for the agent.",
    )

    request_metadata: Optional[RequestMetadata] = Field(
        default=None,
        description="The metadata for the request.",
    )

    model_config = ConfigDict(extra="allow")
