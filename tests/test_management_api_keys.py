# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.api_keys`` read tests -- SecretStr redaction is the focus."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.management._redaction import (  # noqa: E402
    SecretRedactionFilter,
    get_management_logger,
    mask_secret,
)
from dashscope.management.types import ApiKey  # noqa: E402
from dashscope.transport._response import APIResponse  # noqa: E402

from tests.management_helpers import (  # noqa: E402
    action_handler,
    make_async_client,
    make_client,
    ms_body,
)

SECRET = "sk-PLAINTEXTSECRET123456"
BODIES = {
    "ListApiKeys": ms_body(
        apiKeys=[
            {"apiKeyId": 1, "apiKeyValue": SECRET, "workspaceId": "ws-1"},
        ],
        totalCount=1,
    ),
    "GetApiKey": ms_body(
        apiKey={"apiKeyId": 1, "apiKeyValue": SECRET, "description": "demo"},
    ),
}


def test_list_typed_and_masked_repr():
    client = make_client(action_handler(BODIES))
    page = client.api_keys.list()
    key = page.items[0]
    assert isinstance(key, ApiKey)
    assert key.api_key_id == 1
    assert SECRET not in repr(key)
    assert SECRET not in str(key)
    client.close()


def test_get_path_substitution_and_masking():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["GetApiKey"])

    client = make_client(handler)
    key = client.api_keys.get(1)
    assert seen[0].url.path == "/modelstudio/apikeys/1"
    assert seen[0].headers["x-acs-action"] == "GetApiKey"
    assert key.description == "demo"
    assert SECRET not in repr(key)
    client.close()


def test_secret_value_only_via_get_secret_value():
    client = make_client(action_handler(BODIES))
    key = client.api_keys.get(1)
    assert key.api_key_value.get_secret_value() == SECRET
    client.close()


def test_model_dump_masks_secret():
    client = make_client(action_handler(BODIES))
    key = client.api_keys.get(1)
    dumped = json.dumps(key.model_dump(), default=str)
    assert SECRET not in dumped
    assert "**********" in dumped
    client.close()


def test_model_dump_mode_json_masks_secret():
    client = make_client(action_handler(BODIES))
    key = client.api_keys.get(1)
    dumped = key.model_dump(mode="json")
    assert SECRET not in json.dumps(dumped)
    client.close()


def test_logging_model_repr_does_not_leak(caplog):
    client = make_client(action_handler(BODIES))
    key = client.api_keys.get(1)
    logger = logging.getLogger("test.management.apikeys")
    with caplog.at_level(logging.INFO, logger="test.management.apikeys"):
        logger.info("fetched key %r", key)
    assert SECRET not in caplog.text
    client.close()


def test_redaction_filter_masks_free_text_sk_token():
    filt = SecretRedactionFilter()
    record = logging.LogRecord(
        "t",
        logging.INFO,
        __file__,
        1,
        "key is %s",
        (SECRET,),
        None,
    )
    assert filt.filter(record) is True
    assert SECRET not in record.getMessage()
    assert mask_secret(SECRET) in record.getMessage()


def test_mask_secret_short_fully_masked():
    assert mask_secret("short") == "***"
    assert mask_secret(SECRET) == "***" + SECRET[-4:]


def test_management_logger_carries_redaction_filter():
    logger = get_management_logger("management.test")
    assert any(isinstance(f, SecretRedactionFilter) for f in logger.filters)
    # Idempotent: a second fetch must not stack duplicate filters.
    again = get_management_logger("management.test")
    assert again is logger
    assert (
        sum(isinstance(f, SecretRedactionFilter) for f in logger.filters) == 1
    )


def test_management_logger_masks_short_sk_token(caplog):
    # 11 chars: below the core filter's 16-char threshold, so only the
    # management filter stands between this token and the log line.
    short_key = "sk-short123"
    logger = get_management_logger("management.test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("using key %s", short_key)
    assert short_key not in caplog.text
    assert mask_secret(short_key) in caplog.text


def test_redaction_filter_masks_sk_token_at_any_boundary():
    # Short keys adjacent to '=' or embedded in JSON must not leak;
    # space-split masking only caught whitespace-delimited tokens.
    short_key = "sk-short123"
    filt = SecretRedactionFilter()
    for template in ("using key=%s", 'body: {"apiKey": "%s"}'):
        record = logging.LogRecord(
            "t",
            logging.INFO,
            __file__,
            1,
            template,
            (short_key,),
            None,
        )
        assert filt.filter(record) is True
        assert short_key not in record.getMessage()
        assert mask_secret(short_key) in record.getMessage()


def test_redaction_filter_leaves_sk_lookalikes_alone():
    # "task-123456" embeds "sk-123456" but is not a key token.
    filt = SecretRedactionFilter()
    record = logging.LogRecord(
        "t",
        logging.INFO,
        __file__,
        1,
        "finished task-123456 ok",
        (),
        None,
    )
    assert filt.filter(record) is True
    assert record.getMessage() == "finished task-123456 ok"


def test_raw_overload():
    client = make_client(action_handler(BODIES))
    resp = client.api_keys.get(1, raw=True)
    assert isinstance(resp, APIResponse)
    # The raw body still carries the plaintext -- caller's responsibility.
    assert resp.http_response.json()["apiKey"]["apiKeyValue"] == SECRET
    client.close()


async def test_async_list_and_get_masked():
    client = make_async_client(action_handler(BODIES))
    page = await client.api_keys.list()
    assert SECRET not in repr(page.items[0])
    key = await client.api_keys.get(1)
    assert key.api_key_value.get_secret_value() == SECRET
    assert SECRET not in repr(key)
    await client.aclose()
