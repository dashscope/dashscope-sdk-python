# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Live e2e for the read-only ManagementClient surface (G1/G2/G5).

Skipped unless RAM credentials are exported. Only public read ops are
exercised -- ListWorkspaces, ListModels, GetBillingOverview and
GetTokenPlanAccountDetail -- so a run never creates, mutates or deletes
anything. The point is to prove the V3 ACS3-HMAC-SHA256 signature (with
the ``management2`` sdk headers folded in) is accepted by the real
``modelstudio.{region}.aliyuncs.com`` gateway and that the dual envelope
parses into typed models::

    ALIBABA_CLOUD_ACCESS_KEY_ID=... ALIBABA_CLOUD_ACCESS_KEY_SECRET=... \\
        DASHSCOPE_MGMT_REGION=cn-beijing \\
        pytest tests/protocols/test_management_e2e.py

``DASHSCOPE_MGMT_AK`` / ``DASHSCOPE_MGMT_SK`` (and ``DASHSCOPE_MGMT_STS_TOKEN``
for STS) are accepted as alternatives to the standard aliyun env vars. A
business error (account not enrolled in TokenPlan, no billing data, RAM
permission) skips the case rather than failing it; only a rejected
signature -- the one thing the SDK controls -- fails loud.
"""

from __future__ import annotations

import os

import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.management import (  # noqa: E402
    AsyncManagementClient,
    BillingOverview,
    ManagementClient,
    ManagementModel,
    TokenPage,
    TokenPlanAccount,
    Workspace,
)

# A gateway signature rejection means the SDK signed wrong -- a real bug.
# Every other business error is environmental (eligibility / permission),
# so the case skips instead of failing.
_SIGNATURE_ERROR_CODES = frozenset(
    {
        "SignatureDoesNotMatch",
        "InvalidSignature",
        "IncompleteSignature",
    },
)


def _creds():
    ak = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID") or os.environ.get(
        "DASHSCOPE_MGMT_AK",
    )
    sk = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or os.environ.get(
        "DASHSCOPE_MGMT_SK",
    )
    if not ak or not sk:
        pytest.skip("RAM access key id / secret not set")
    token = os.environ.get("ALIBABA_CLOUD_SECURITY_TOKEN") or os.environ.get(
        "DASHSCOPE_MGMT_STS_TOKEN",
    )
    return ak, sk, token


def _region():
    return (
        os.environ.get("DASHSCOPE_MGMT_REGION")
        or os.environ.get("ALIBABA_CLOUD_REGION")
        or "cn-beijing"
    )


def _make_client():
    ak, sk, token = _creds()
    return ManagementClient(
        access_key_id=ak,
        access_key_secret=sk,
        security_token=token,
        region=_region(),
        max_retries=1,
        raise_on_error=False,
    )


def _make_async_client():
    ak, sk, token = _creds()
    return AsyncManagementClient(
        access_key_id=ak,
        access_key_secret=sk,
        security_token=token,
        region=_region(),
        max_retries=1,
        raise_on_error=False,
    )


def _assert_accepted(result):
    """Fail on a rejected signature, skip on other business errors."""
    if result.is_error:
        if result.code in _SIGNATURE_ERROR_CODES:
            pytest.fail(
                f"V3 signature rejected by gateway: "
                f"{result.code} {result.message}",
            )
        pytest.skip(
            f"account not eligible for this read op: "
            f"{result.code} {result.message}",
        )
    assert result.request_id, "gateway response carried no request id"


def test_list_workspaces():
    with _make_client() as client:
        page = client.workspaces.list(max_results=10)
    _assert_accepted(page)
    assert isinstance(page, TokenPage)
    for workspace in page:
        assert isinstance(workspace, Workspace)


def test_list_models():
    with _make_client() as client:
        page = client.workspaces.models.list(max_results=10)
    _assert_accepted(page)
    assert isinstance(page, TokenPage)
    for model in page:
        assert isinstance(model, ManagementModel)


def test_billing_overview():
    with _make_client() as client:
        overview = client.billing.overview()
    _assert_accepted(overview)
    assert isinstance(overview, BillingOverview)


def test_token_plan_account():
    with _make_client() as client:
        account = client.token_plan.account.get()
    _assert_accepted(account)
    assert isinstance(account, TokenPlanAccount)


async def test_async_list_models():
    async with _make_async_client() as client:
        page = await client.workspaces.models.list(max_results=10)
    _assert_accepted(page)
    assert isinstance(page, TokenPage)
    for model in page:
        assert isinstance(model, ManagementModel)
