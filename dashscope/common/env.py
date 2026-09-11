# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re

from dashscope.common.constants import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_API_KEY_FILE_PATH_ENV,
    DASHSCOPE_API_REGION_ENV,
    DASHSCOPE_API_VERSION_ENV,
)
from dashscope.common.error import InputRequired

# workspace_id and region become DNS labels of the MaaS endpoint host, so
# they must not contain characters (/, ?, #, @, :, whitespace, ...) that
# could break out of the host and redirect requests elsewhere.
# \A...\Z rather than ^...$: $ also matches before a trailing newline.
_WORKSPACE_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
# RFC 1123 hostname label: 1-63 chars, no leading/trailing hyphen. Upper
# case is allowed because DNS is case-insensitive and httpx lowercases the
# host, so rejecting it would break working callers for no security gain.
_REGION_PATTERN = re.compile(
    r"\A[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z",
)


def validate_workspace_id(workspace_id):
    """Validate that workspace_id is safe to use as a URL subdomain.

    Raises:
        ValueError: If workspace_id contains characters outside
            [A-Za-z0-9_-], is empty, or exceeds 64 characters.
    """
    if not workspace_id or not _WORKSPACE_ID_PATTERN.fullmatch(
        workspace_id,
    ):
        raise ValueError(
            f"Invalid workspace_id {workspace_id!r}: must match "
            "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        )
    return workspace_id


def validate_region(region):
    """Validate that region is safe to use as a URL subdomain.

    Deliberately a hostname-label check and not a MAAS_REGIONS whitelist:
    rejecting an unknown-but-well-formed region would break callers as soon
    as a subpackage supports a region this list has not caught up with.

    Raises:
        ValueError: If region is not a valid hostname label.
    """
    if not region or not _REGION_PATTERN.fullmatch(region):
        raise ValueError(
            f"Invalid region {region!r}: must be a hostname label "
            "(1-63 chars of [A-Za-z0-9-], not starting or ending "
            "with '-')",
        )
    return region


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
    if _default_workspace_id:
        validate_workspace_id(_default_workspace_id)
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
        validate_workspace_id(ws)
        return url.replace("{workspace_id}", ws)
    raise InputRequired(
        "The base URL contains '{workspace_id}' but no workspace id "
        "is available: pass workspace to the API call, set the "
        "DASHSCOPE_WORKSPACE_ID environment variable, or call "
        "dashscope.set_region(region, workspace_id).",
    )
