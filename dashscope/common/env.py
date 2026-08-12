# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from dashscope.common.constants import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_API_KEY_FILE_PATH_ENV,
    DASHSCOPE_API_REGION_ENV,
    DASHSCOPE_API_VERSION_ENV,
)

api_region = os.environ.get(DASHSCOPE_API_REGION_ENV, "cn-beijing")
api_version = os.environ.get(DASHSCOPE_API_VERSION_ENV, "v1")
# read the api key from env
api_key = os.environ.get(DASHSCOPE_API_KEY_ENV)
api_key_file_path = os.environ.get(DASHSCOPE_API_KEY_FILE_PATH_ENV)

# define api base url, ensure end /
base_http_api_url = os.environ.get(
    "DASHSCOPE_HTTP_BASE_URL",
    f"https://dashscope.aliyuncs.com/api/{api_version}",
)
base_websocket_api_url = os.environ.get(
    "DASHSCOPE_WEBSOCKET_BASE_URL",
    f"wss://dashscope.aliyuncs.com/api-ws/{api_version}/inference",
)
base_compatible_api_url = os.environ.get(
    "DASHSCOPE_COMPATIBLE_BASE_URL",
    f"https://dashscope.aliyuncs.com/compatible-mode/{api_version}",
)

# HTTP connection pool configuration
# Can be set via environment variable or module-level variable
# Example:
#   import dashscope
#   dashscope.http_connection_pool_size = 50
# Or:
#   export DASHSCOPE_HTTP_CONNECTION_POOL_SIZE=50
http_connection_pool_size = int(
    os.environ.get("DASHSCOPE_HTTP_CONNECTION_POOL_SIZE", "20"),
)
