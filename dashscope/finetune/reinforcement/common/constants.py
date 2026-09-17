# -*- coding: utf-8 -*-
import logging
import os


# --------------------------
# Environment Configuration
# --------------------------


def get_bool_env(env_var: str, default: bool) -> bool:
    """Safely parse boolean environment variable."""
    value = os.getenv(env_var, str(default)).strip().lower()
    return value in ("true", "1", "yes")


def get_int_env(env_var: str, default: int) -> int:
    """Safely parse integer environment variable."""
    try:
        return int(os.getenv(env_var, str(default)))
    except ValueError:
        return default


# Base URL configuration
DASHSCOPE_HTTP_BASE_URL = os.environ.get(
    "DASHSCOPE_HTTP_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1",
)

# API Key configuration
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
FC_API_KEY = os.environ.get("FC_API_KEY", DASHSCOPE_API_KEY)

# --------------------------
# Logging Configuration
# --------------------------
LOGGER_NAME = os.environ.get("LOGGER_NAME", "dashscope")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
LOGGER_FILTER_FIELDS = {
    "trigger_token",
    "instance_token",
    "api_token",
    "api_key",
    "password",
    "access_key",
    "secret_key",
}  # Case-sensitive field names to filter

# Initialize logger
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)

# --------------------------
# Service Configuration
# --------------------------
HTTP_REQUEST_TIMEOUT = get_int_env("HTTP_REQUEST_TIMEOUT", 600)

# Bailian File Service
BAILIAN_FILE_API = os.environ.get(
    "BAILIAN_FILE_API",
    f"{DASHSCOPE_HTTP_BASE_URL}/files",
)
BAILIAN_FILE_TIMEOUT = get_int_env("BAILIAN_FILE_TIMEOUT", 600)

# Function Compute Endpoints
FC_BASE_ENDPOINT = f"{DASHSCOPE_HTTP_BASE_URL}/fine-tunes"
FC_UPLOAD_OSS_API = os.environ.get(
    "FC_UPLOAD_OSS_API",
    f"{FC_BASE_ENDPOINT}/generate/faas/upload/url",
)
FC_REGISTER_ROLLOUT_API = os.environ.get(
    "FC_REGISTER_ROLLOUT_API",
    f"{FC_BASE_ENDPOINT}/create/rollout",
)
FC_REGISTER_REWARD_API = os.environ.get(
    "FC_REGISTER_REWARD_API",
    f"{FC_BASE_ENDPOINT}/create/reward",
)
FC_REGISTER_GROUP_REWARD_API = os.environ.get(
    "FC_REGISTER_GROUP_REWARD_API",
    f"{FC_BASE_ENDPOINT}/create/group-reward",
)
FC_LOAD_API = os.environ.get("FC_LOAD_API", f"{FC_BASE_ENDPOINT}/online/faas")
FC_QUERY_API = os.environ.get("FC_QUERY_API", f"{FC_BASE_ENDPOINT}/query/faas")
FC_QUERY_LOG_API = os.environ.get(
    "FC_QUERY_LOG_API",
    f"{FC_BASE_ENDPOINT}/query/faas/log",
)
FC_DELETE_API = os.environ.get(
    "FC_DELETE_API",
    f"{FC_BASE_ENDPOINT}/delete/faas",
)
FC_LAYER_CREATE_API = os.environ.get(  # POST
    "FC_LAYER_CREATE_API",
    f"{FC_BASE_ENDPOINT}/create/faas/layer",
)
FC_LAYER_QUERY_API = os.environ.get(  # GET
    "FC_LAYER_QUERY_API",
    f"{FC_BASE_ENDPOINT}/query/faas/layer/status",
)

# --------------------------
# Function Compute Runtime
# --------------------------
FC_FILES_START = "start.sh"
# FC_PYPI_LIB = os.environ.get("FC_PYPI_LIB", "dashscope")
FC_PYPI_LIB = os.environ.get("FC_PYPI_LIB", "")
FC_PYPI_REPO = os.environ.get(
    "FC_PYPI_REPO",
    "https://mirrors.aliyun.com/pypi/simple/",
)
FC_SERVER_CLASSPATH = os.environ.get(
    "FC_SERVER_CLASSPATH",
    "dashscope.finetune.reinforcement.component.server.server",
)
FC_WORKERS_COUNT = get_int_env("FC_WORKERS_COUNT", 2)
FC_LAYER_USED = get_bool_env("FC_LAYER_USED", True)
FC_LAYER_NAME = "fc_layer"
FC_REQUIREMENTS_FILE = "./requirements.txt"
FC_ZIP_EXCLUDE_PATTERNS = os.environ.get(
    "FC_ZIP_EXCLUDE_PATTERNS",
    ".git,.gitignore,.DS_Store,.vscode,.Python,.env,.venv,.idea,__pycache__,"
    "*.swp,*.egg,*.egg-info,*.pyc,*.md,*.log,*.tmp,*.bak,"
    "build,develop-eggs,.eggs,test,tests,tmp,temp,data",
)
FC_OSS_FILE_SIZE_WARNING = get_int_env(
    "FC_OSS_FILE_SIZE_WARNING",
    200 * 1024 * 1024,
)  # 200M

# --------------------------
# Datasets Configuration
# --------------------------
DATASETS_FILE_SIZE_WARNING = get_int_env(
    "DATASETS_FILE_SIZE_WARNING",
    1024 * 1024 * 1024,
)  # 1G

# --------------------------
# Tuning Configuration
# --------------------------
TUNING_MODE_NAME = "reinforcement"
ENABLE_TRAJECTORY = get_bool_env("ENABLE_TRAJECTORY", True)
