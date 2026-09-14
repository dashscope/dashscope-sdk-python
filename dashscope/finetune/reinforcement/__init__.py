# -*- coding: utf-8 -*-
try:
    import pydantic  # noqa: F401
    import tenacity  # noqa: F401
    import yaml  # noqa: F401
except ImportError as _rl_dep_err:
    raise ImportError(
        "Agentic RL fine-tuning needs optional dependencies. "
        "Install them with: pip install 'dashscope[rl]'",
    ) from _rl_dep_err

from dashscope.finetune.reinforcement.common.constants import (
    BAILIAN_FILE_API,
    BAILIAN_FILE_TIMEOUT,
    DASHSCOPE_API_KEY,
    DASHSCOPE_HTTP_BASE_URL,
    LOG_LEVEL,
    FC_API_KEY,
    FC_FILES_START,
    FC_LOAD_API,
    FC_PYPI_LIB,
    FC_PYPI_REPO,
    FC_QUERY_API,
    FC_REGISTER_REWARD_API,
    FC_REGISTER_ROLLOUT_API,
    FC_REGISTER_GROUP_REWARD_API,
    FC_SERVER_CLASSPATH,
    FC_UPLOAD_OSS_API,
    FC_WORKERS_COUNT,
    FC_LAYER_CREATE_API,
    FC_LAYER_QUERY_API,
    FC_LAYER_USED,
    FC_LAYER_NAME,
    FC_REQUIREMENTS_FILE,
    FC_ZIP_EXCLUDE_PATTERNS,
    FC_OSS_FILE_SIZE_WARNING,
    DATASETS_FILE_SIZE_WARNING,
    HTTP_REQUEST_TIMEOUT,
    LOGGER_FILTER_FIELDS,
)
from dashscope.finetune.reinforcement.common.errors import (
    AgenticRLError,
    InputError,
    OutputError,
    BaseConnectionError,
    OSSConnectionError,
    OSSUploadError,
    DeploymentError,
    RegistrationError,
    FunctionLoadError,
    InstanceWarmupError,
    InstanceQueryError,
    FunctionLayerError,
    ValidationError,
    ConfigurationError,
    BasePermissionError,
    IOErrorWithCode,
    RuntimeErrorWithCode,
    ValueErrorWithCode,
)
from dashscope.finetune.reinforcement.common.log import logger
from dashscope.finetune.reinforcement.common.model import (
    FunctionComponentModel,
    TuningModel,
    FoundationModel,
    FunctionComponentRuntime,
    AgenticRLFunctionComponent,
    RolloutFunctionComponent,
    RewardFunctionComponent,
    GroupRewardFunctionComponent,
    Datasets,
    Dataset,
    TrainingDataset,
    ValidationDataset,
    MountStorage,
    Training,
    Observability,
    AgenticRLTuning,
    Models,
)
from dashscope.finetune.reinforcement.common.model_types import (
    FileSpec,
    FunctionType,
    DatasetsType,
    DataSourceType,
    RequestFC,
    ResponseFC,
    Status,
    StatusType,
    TrainingType,
)
from dashscope.finetune.reinforcement.common.utils import (
    check_file,
    client_fc,
    create_deployment_files,
    deep_mask,
    generate_random_id,
    get_filepath_classname,
    upload_zip_to_oss_and_by_signed_url,
    set_api_key,
    to_bailian_data,
    zip_dir,
    get_func_type_id,
    deep_remove_none,
    extract_reward_weights,
    get_weights_from_file,
    serialize_for_output,
)
from dashscope.finetune.reinforcement.component import (
    reward_func,
    sub_reward_func,
    aggregate_func,
)
from dashscope.finetune.reinforcement.component.data import (
    AgentOutput,
    Reward,
    RewardInput,
    RewardOutput,
    RolloutInput,
    RolloutOutput,
    GroupReward,
    GroupRewardInput,
    GroupRewardOutput,
    TaskStatus,
)
from dashscope.finetune.reinforcement.component.func_decorator import (
    RewardProcessorMeta,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRolloutProcessor,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRewardProcessor,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractGroupRewardProcessor,
)


def __getattr__(name):
    if name == "app":
        from dashscope.cli.agentic_rl import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# pylint: disable=undefined-all-variable
__all__ = [
    # Constants
    "BAILIAN_FILE_API",
    "BAILIAN_FILE_TIMEOUT",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_HTTP_BASE_URL",
    "LOG_LEVEL",
    "FC_API_KEY",
    "FC_FILES_START",
    "FC_LOAD_API",
    "FC_PYPI_LIB",
    "FC_PYPI_REPO",
    "FC_QUERY_API",
    "FC_REGISTER_REWARD_API",
    "FC_REGISTER_ROLLOUT_API",
    "FC_REGISTER_GROUP_REWARD_API",
    "FC_SERVER_CLASSPATH",
    "FC_UPLOAD_OSS_API",
    "FC_WORKERS_COUNT",
    "FC_LAYER_CREATE_API",
    "FC_LAYER_QUERY_API",
    "FC_LAYER_USED",
    "FC_LAYER_NAME",
    "FC_REQUIREMENTS_FILE",
    "FC_ZIP_EXCLUDE_PATTERNS",
    "FC_OSS_FILE_SIZE_WARNING",
    "DATASETS_FILE_SIZE_WARNING",
    "HTTP_REQUEST_TIMEOUT",
    "LOGGER_FILTER_FIELDS",
    # Errors
    "AgenticRLError",
    "InputError",
    "OutputError",
    "BaseConnectionError",
    "OSSConnectionError",
    "OSSUploadError",
    "DeploymentError",
    "RegistrationError",
    "FunctionLoadError",
    "FunctionLayerError",
    "InstanceWarmupError",
    "InstanceQueryError",
    "ValidationError",
    "ConfigurationError",
    "BasePermissionError",
    "IOErrorWithCode",
    "RuntimeErrorWithCode",
    "ValueErrorWithCode",
    # Logging
    "logger",
    # Utilities
    "check_file",
    "client_fc",
    "create_deployment_files",
    "deep_mask",
    "generate_random_id",
    "get_filepath_classname",
    "set_api_key",
    "to_bailian_data",
    "upload_zip_to_oss_and_by_signed_url",
    "zip_dir",
    "get_func_type_id",
    "deep_remove_none",
    "extract_reward_weights",
    "get_weights_from_file",
    "serialize_for_output",
    # Model Types
    "FileSpec",
    "FunctionType",
    "DatasetsType",
    "DataSourceType",
    "RequestFC",
    "ResponseFC",
    "Status",
    "StatusType",
    "TrainingType",
    # Data models
    "RolloutInput",
    "RolloutOutput",
    "RewardInput",
    "RewardOutput",
    "GroupRewardInput",
    "GroupRewardOutput",
    "GroupReward",
    "Reward",
    "AgentOutput",
    "TaskStatus",
    # Func Manager
    "reward_func",
    "sub_reward_func",
    "aggregate_func",
    "RewardProcessorMeta",
    "AbstractRolloutProcessor",
    "AbstractRewardProcessor",
    "AbstractGroupRewardProcessor",
    # Core Models
    "FunctionComponentModel",
    "TuningModel",
    "FoundationModel",
    "FunctionComponentRuntime",
    "AgenticRLFunctionComponent",
    "RolloutFunctionComponent",
    "RewardFunctionComponent",
    "GroupRewardFunctionComponent",
    "Datasets",
    "Dataset",
    "TrainingDataset",
    "ValidationDataset",
    "MountStorage",
    "Training",
    "Observability",
    "AgenticRLTuning",
    "Models",
    "app",
]
