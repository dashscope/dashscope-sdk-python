# -*- coding: utf-8 -*-
# Public names are provided lazily via __getattr__; __all__ entries are not
# bindings.
# pylint: disable=undefined-all-variable
# -------------------------------------------------------------------------- #
#                               Base Components                              #
# -------------------------------------------------------------------------- #
from .data.base_data_model import (
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
from .data.reward_input import (
    RewardInput,
)
from .data.reward_output import (
    Reward,
    RewardOutput,
)
from .data.rollout_input import (
    RolloutInput,
)
from .data.rollout_output import (
    RolloutOutput,
)

# -------------------------------------------------------------------------- #
#                                   Demos                                    #
# -------------------------------------------------------------------------- #
from .demo import (
    DemoRewardProcessor,
    DemoRolloutProcessor,
)
from .func_decorator import (
    reward_func,
    sub_reward_func,
    aggregate_func,
)

# -------------------------------------------------------------------------- #
#                              Func Manager                                  #
# -------------------------------------------------------------------------- #
from .func_manager import (
    FuncManager,
)

# -------------------------------------------------------------------------- #
#                                   Observability                            #
# -------------------------------------------------------------------------- #
# pylint: disable=no-name-in-module
from .observability import (
    observe_processor,
)
from .parser.base_parser import (
    BaseRequestParser,
)

# -------------------------------------------------------------------------- #
#                                  Parsers                                   #
# -------------------------------------------------------------------------- #
from .parser.reward_parser import (
    RewardRequestParser,
)
from .parser.rollout_parser import (
    RolloutRequestParser,
)

# -------------------------------------------------------------------------- #
#                                Processors                                  #
# -------------------------------------------------------------------------- #
from .processor import (
    AbstractRewardProcessor,
)
from .processor import (
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
