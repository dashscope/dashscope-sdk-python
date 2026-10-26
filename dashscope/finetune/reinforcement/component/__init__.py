# -*- coding: utf-8 -*-
# Public names are provided lazily via __getattr__; __all__ entries are not
# bindings.
# pylint: disable=undefined-all-variable
# -------------------------------------------------------------------------- #
#                               Base Components                              #
# -------------------------------------------------------------------------- #
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
    TaskStatus,
    ModelProtocol,
    Task,
    Resource,
    AgentOutput,
)

# -------------------------------------------------------------------------- #
#                                 Data Models                                #
# -------------------------------------------------------------------------- #
from dashscope.finetune.reinforcement.component.data.reward_input import (
    RewardInput,
)
from dashscope.finetune.reinforcement.component.data.reward_output import (
    Reward,
    RewardOutput,
)
from dashscope.finetune.reinforcement.component.data.rollout_input import (
    RolloutInput,
)
from dashscope.finetune.reinforcement.component.data.rollout_output import (
    RolloutOutput,
)

# -------------------------------------------------------------------------- #
#                                   Demos                                    #
# -------------------------------------------------------------------------- #
from dashscope.finetune.reinforcement.component.demo import (
    DemoRewardProcessor,
    DemoRolloutProcessor,
)
from dashscope.finetune.reinforcement.component.func_decorator import (
    reward_func,
    sub_reward_func,
    aggregate_func,
)

# -------------------------------------------------------------------------- #
#                              Func Manager                                  #
# -------------------------------------------------------------------------- #
from dashscope.finetune.reinforcement.component.func_manager import (
    FuncManager,
)

# -------------------------------------------------------------------------- #
#                                   Observability                            #
# -------------------------------------------------------------------------- #
# pylint: disable=no-name-in-module
from dashscope.finetune.reinforcement.component.observability import (
    observe_processor,
)
from dashscope.finetune.reinforcement.component.parser.base_parser import (
    BaseRequestParser,
)

# -------------------------------------------------------------------------- #
#                                  Parsers                                   #
# -------------------------------------------------------------------------- #
from dashscope.finetune.reinforcement.component.parser.reward_parser import (
    RewardRequestParser,
)
from dashscope.finetune.reinforcement.component.parser.rollout_parser import (
    RolloutRequestParser,
)

# -------------------------------------------------------------------------- #
#                                Processors                                  #
# -------------------------------------------------------------------------- #
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRewardProcessor,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRolloutProcessor,
)

__all__ = [
    # Base
    "BaseRequestParser",
    "BaseDataModel",
    "TaskStatus",
    "ModelProtocol",
    "Task",
    "Resource",
    "AgentOutput",
    # Func Manager
    "FuncManager",
    "reward_func",
    "sub_reward_func",
    "aggregate_func",
    # Data Models
    "RewardInput",
    "RolloutInput",
    "Reward",
    "RewardOutput",
    "RolloutOutput",
    # Parsers
    "RewardRequestParser",
    "RolloutRequestParser",
    # Processors
    "AbstractRewardProcessor",
    "AbstractRolloutProcessor",
    # Demos
    "DemoRewardProcessor",
    "DemoRolloutProcessor",
    # Observability
    "observe_processor",
]
