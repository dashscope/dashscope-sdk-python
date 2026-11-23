# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""ManagementClient -- the RAM OpenAPI (ModelStudio/2026-02-10) surface.

An optional extra: ``pip install 'dashscope[management]'`` pulls in
``alibabacloud_credentials`` (the multi-tenant credential chain) and
``alibabacloud_openapi_util`` (ACS3-HMAC-SHA256 V3 signing). The core
``dashscope`` package does not import this module, so installing the base
SDK never forces the aliyun cloud libraries.

This revision is read-only (G1/G2/G5): workspace / API-key / billing /
TokenPlan reads. Write operations are high-risk and deferred.
"""

from __future__ import annotations

try:
    import alibabacloud_credentials  # noqa: F401
    import alibabacloud_openapi_util  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "dashscope.management requires the optional 'management' extra "
        "(alibabacloud_credentials, alibabacloud_openapi_util). "
        "Install it with: pip install 'dashscope[management]'",
    ) from exc

from dashscope.management._credentials import (
    CredentialResolver,
    ResolvedCredential,
)
from dashscope.management._errors import (
    Envelope,
    check_management_response,
    decode_body,
    management_error,
    read_envelope,
)
from dashscope.management._pagination import TokenPage
from dashscope.management._redaction import (
    REDACTED,
    SecretRedactionFilter,
    mask_secret,
    redact_body,
)
from dashscope.management.client import (
    AsyncManagementClient,
    ManagementClient,
    ManagementConfig,
)
from dashscope.management.types import (
    ApiKey,
    ApiKeyAuth,
    BillingGroup,
    BillingOverview,
    BillingTrend,
    CostTotals,
    GroupByTotal,
    InferenceMetadata,
    ManagementModel,
    ModelAccessScope,
    ModelInfo,
    ModelLimit,
    ModelPermission,
    OrgMember,
    OrgMemberPage,
    OrgMembership,
    Permissions,
    Price,
    RateLimit,
    ResultByTime,
    Seat,
    SeatPage,
    SeatStats,
    SharedPackage,
    SharedPackagePage,
    SubscriptionSeatItem,
    SubscriptionStats,
    TokenPlanAccount,
    Workspace,
)

__all__ = [
    "REDACTED",
    "ApiKey",
    "ApiKeyAuth",
    "AsyncManagementClient",
    "BillingGroup",
    "BillingOverview",
    "BillingTrend",
    "CostTotals",
    "CredentialResolver",
    "Envelope",
    "GroupByTotal",
    "InferenceMetadata",
    "ManagementClient",
    "ManagementConfig",
    "ManagementModel",
    "ModelAccessScope",
    "ModelInfo",
    "ModelLimit",
    "ModelPermission",
    "OrgMember",
    "OrgMemberPage",
    "OrgMembership",
    "Permissions",
    "Price",
    "RateLimit",
    "ResolvedCredential",
    "ResultByTime",
    "Seat",
    "SeatPage",
    "SeatStats",
    "SecretRedactionFilter",
    "SharedPackage",
    "SharedPackagePage",
    "SubscriptionSeatItem",
    "SubscriptionStats",
    "TokenPage",
    "TokenPlanAccount",
    "Workspace",
    "check_management_response",
    "decode_body",
    "management_error",
    "mask_secret",
    "read_envelope",
    "redact_body",
]
