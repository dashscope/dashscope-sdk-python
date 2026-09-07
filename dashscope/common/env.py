# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from dashscope.common.constants import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_API_KEY_FILE_PATH_ENV,
    DASHSCOPE_API_REGION_ENV,
    DASHSCOPE_API_VERSION_ENV,
)
from dashscope.common.error import InputRequired

# MaaS regions: region -> URL subdomain identifier
# cn-beijing is deliberately excluded: it is the default
# DASHSCOPE_API_REGION and must keep using the legacy
# dashscope.aliyuncs.com endpoints. cn-beijing MaaS URLs are only
# reachable via an explicit dashscope.set_region() call.
MAAS_REGIONS = {
    "ap-southeast-1": "ap-southeast-1",
    "us-east-1": "us-east-1",
    "cn-hongkong": "cn-hongkong",
    "eu-central-1": "eu-central-1",
    "ap-northeast-1": "ap-northeast-1",
}

api_region = os.environ.get(DASHSCOPE_API_REGION_ENV, "cn-beijing")
api_version = os.environ.get(DASHSCOPE_API_VERSION_ENV, "v1")
# read the api key from env
api_key = os.environ.get(DASHSCOPE_API_KEY_ENV)
api_key_file_path = os.environ.get(DASHSCOPE_API_KEY_FILE_PATH_ENV)

# Optional default workspace id from environment (used to resolve MaaS URLs)
_default_workspace_id = os.environ.get("DASHSCOPE_WORKSPACE_ID")

# define api base url, ensure end /
if api_region in MAAS_REGIONS:
    _maas_region_id = MAAS_REGIONS[api_region]
    _ws = _default_workspace_id if _default_workspace_id else "{workspace_id}"
    _maas_host = f"{_ws}.{_maas_region_id}.maas.aliyuncs.com"
    base_http_api_url = os.environ.get(
        "DASHSCOPE_HTTP_BASE_URL",
        f"https://{_maas_host}/api/{api_version}",
    )
    base_websocket_api_url = os.environ.get(
        "DASHSCOPE_WEBSOCKET_BASE_URL",
        f"wss://{_maas_host}/api-ws/{api_version}/inference",
    )
    base_compatible_api_url = os.environ.get(
        "DASHSCOPE_COMPATIBLE_BASE_URL",
        f"https://{_maas_host}/compatible-mode/{api_version}",
    )
else:
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


def resolve_base_url(url, workspace_id=None):
    """Resolve {workspace_id} placeholder in base URL.

    Args:
        url: The base URL, possibly containing {workspace_id}.
        workspace_id: The workspace id to substitute.

    Returns:
        The resolved URL string.

    Raises:
        InputRequired: If the URL contains {workspace_id} but no
            workspace id is available.
    """
    if "{workspace_id}" not in url:
        return url
    ws = workspace_id or _default_workspace_id
    if ws:
        return url.replace("{workspace_id}", ws)
    raise InputRequired(
        "The base URL contains '{workspace_id}' but no workspace id "
        "is available: pass workspace to the API call, set the "
        "DASHSCOPE_WORKSPACE_ID environment variable, or call "
        "dashscope.set_region(region, workspace_id).",
    )
