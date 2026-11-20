# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Public response models for the management read surface.

Every model is permissive (``extra="allow"``, optional fields with
defaults) so a gateway that adds or renames a field degrades rather than
raises. Wire names are camelCase on the modelstudio ops and PascalCase on
the tokenplan ops; ``Field(validation_alias=...)`` maps each to a
snake_case attribute, and ``populate_by_name=True`` keeps the snake name
usable too.

``ApiKey.api_key_value`` is a ``SecretStr``: even though these are read
ops, the gateway returns the plaintext key, so it is masked in ``repr``,
``model_dump`` and logs. ``OrgMember.masked_api_key`` is already masked
server-side and stays a plain string.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from dashscope.types.common import ResponseBase
from dashscope.types.pagination import BasePage

__all__ = [
    "ApiKey",
    "ApiKeyAuth",
    "BillingGroup",
    "BillingOverview",
    "BillingTrend",
    "CostTotals",
    "GroupByTotal",
    "InferenceMetadata",
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
    "ResultByTime",
    "Seat",
    "SeatPage",
    "SeatStats",
    "SharedPackage",
    "SharedPackagePage",
    "SubscriptionSeatItem",
    "SubscriptionStats",
    "TokenPlanAccount",
    "Workspace",
]


class _Model(BaseModel):
    """Permissive base for nested payload models (no envelope)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class _Response(ResponseBase):
    """Permissive base for top-level responses (carries the envelope)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# --- workspaces / models -------------------------------------------------


class Workspace(_Model):
    api_host: Optional[str] = Field(None, validation_alias="apiHost")
    gmt_create: Optional[int] = Field(None, validation_alias="gmtCreate")
    region: Optional[str] = None
    service_site: Optional[str] = Field(None, validation_alias="serviceSite")
    workspace_id: Optional[str] = Field(None, validation_alias="workspaceId")
    workspace_name: Optional[str] = Field(
        None,
        validation_alias="workspaceName",
    )


class InferenceMetadata(_Model):
    request_modality: Optional[Any] = Field(
        None,
        validation_alias="requestModality",
    )
    response_modality: Optional[Any] = Field(
        None,
        validation_alias="responseModality",
    )


class ModelInfo(_Model):
    context_window: Optional[int] = Field(
        None,
        validation_alias="contextWindow",
    )
    max_input_tokens: Optional[int] = Field(
        None,
        validation_alias="maxInputTokens",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        validation_alias="maxOutputTokens",
    )
    max_reasoning_tokens: Optional[int] = Field(
        None,
        validation_alias="maxReasoningTokens",
    )
    reasoning_max_input_tokens: Optional[int] = Field(
        None,
        validation_alias="reasoningMaxInputTokens",
    )
    reasoning_max_output_tokens: Optional[int] = Field(
        None,
        validation_alias="reasoningMaxOutputTokens",
    )


class Price(_Model):
    """A model price entry; fields vary by provider, kept as extras."""


class ManagementModel(_Model):
    model: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    inference_metadata: Optional[InferenceMetadata] = Field(
        None,
        validation_alias="inferenceMetadata",
    )
    prices: List[Price] = Field(default_factory=list)
    published_time: Optional[Any] = Field(
        None,
        validation_alias="publishedTime",
    )
    model_info: Optional[ModelInfo] = Field(None, validation_alias="modelInfo")
    provider: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)


class RateLimit(_Model):
    request_limit: Optional[int] = Field(None, validation_alias="requestLimit")
    request_limit_period: Optional[str] = Field(
        None,
        validation_alias="requestLimitPeriod",
    )
    usage_limit: Optional[int] = Field(None, validation_alias="usageLimit")
    usage_limit_field: Optional[str] = Field(
        None,
        validation_alias="usageLimitField",
    )
    usage_limit_period: Optional[str] = Field(
        None,
        validation_alias="usageLimitPeriod",
    )
    async_user_queue_limit: Optional[int] = Field(
        None,
        validation_alias="asyncUserQueueLimit",
    )
    async_user_concurrency_limit: Optional[int] = Field(
        None,
        validation_alias="asyncUserConcurrencyLimit",
    )


class ModelLimit(_Model):
    model: Optional[str] = None
    name: Optional[str] = None
    workspace_id: Optional[str] = Field(None, validation_alias="workspaceId")
    model_limit: Optional[RateLimit] = Field(
        None,
        validation_alias="modelLimit",
    )
    workspace_limit: Optional[RateLimit] = Field(
        None,
        validation_alias="workspaceLimit",
    )


class Permissions(_Model):
    inference: Optional[bool] = None
    fine_tune: Optional[bool] = Field(None, validation_alias="fineTune")
    deploy: Optional[bool] = None


class ModelPermission(_Model):
    model: Optional[str] = None
    name: Optional[str] = None
    permissions: Optional[Permissions] = None


# --- api keys ------------------------------------------------------------


class ModelAccessScope(_Model):
    accessible_models: List[str] = Field(
        default_factory=list,
        validation_alias="accessibleModels",
    )
    allow_all_models: Optional[bool] = Field(
        None,
        validation_alias="allowAllModels",
    )


class ApiKeyAuth(_Model):
    access_ips: List[str] = Field(
        default_factory=list,
        validation_alias="accessIps",
    )
    model_access_scope: Optional[ModelAccessScope] = Field(
        None,
        validation_alias="modelAccessScope",
    )
    type: Optional[str] = None


class ApiKey(_Model):
    api_key_id: Optional[int] = Field(None, validation_alias="apiKeyId")
    api_key_value: Optional[SecretStr] = Field(
        None,
        validation_alias="apiKeyValue",
    )
    auth: Optional[ApiKeyAuth] = None
    created_by: Optional[str] = Field(None, validation_alias="createdBy")
    description: Optional[str] = None
    disabled: Optional[int] = None
    gmt_create: Optional[int] = Field(None, validation_alias="gmtCreate")
    workspace_id: Optional[str] = Field(None, validation_alias="workspaceId")
    workspace_name: Optional[str] = Field(
        None,
        validation_alias="workspaceName",
    )


# --- billing -------------------------------------------------------------


class BillingGroup(_Model):
    key: Optional[str] = None
    name: Optional[str] = None
    article_codes: List[str] = Field(
        default_factory=list,
        validation_alias="articleCodes",
    )
    amount: Optional[str] = None
    percentage: Optional[str] = None


class BillingOverview(_Response):
    currency: Optional[str] = None
    total_amount: Optional[str] = Field(None, validation_alias="totalAmount")
    pretax_amount: Optional[str] = Field(None, validation_alias="pretaxAmount")
    tax_amount: Optional[str] = Field(None, validation_alias="taxAmount")
    groups: List[BillingGroup] = Field(default_factory=list)


class CostTotals(_Model):
    amount: Optional[str] = None
    pretax_amount: Optional[str] = Field(None, validation_alias="pretaxAmount")
    tax_amount: Optional[str] = Field(None, validation_alias="taxAmount")
    currency: Optional[str] = None


class GroupByTotal(_Model):
    key: Optional[str] = None
    name: Optional[str] = None
    amount: Optional[str] = None
    pretax_amount: Optional[str] = Field(None, validation_alias="pretaxAmount")
    tax_amount: Optional[str] = Field(None, validation_alias="taxAmount")


class ResultByTime(_Model):
    period: Optional[str] = None
    total: Optional[Any] = None
    period_details: Optional[Any] = Field(
        None,
        validation_alias="periodDetails",
    )


class BillingTrend(_Response):
    cost_totals: Optional[CostTotals] = Field(
        None,
        validation_alias="costTotals",
    )
    group_by_total: List[GroupByTotal] = Field(
        default_factory=list,
        validation_alias="groupByTotal",
    )
    result_by_time: List[ResultByTime] = Field(
        default_factory=list,
        validation_alias="resultByTime",
    )
    history_period: Optional[Any] = Field(
        None,
        validation_alias="historyPeriod",
    )


# --- token plan ----------------------------------------------------------


class OrgMembership(_Model):
    org_id: Optional[str] = Field(None, validation_alias="OrgId")
    member_status: Optional[str] = Field(None, validation_alias="MemberStatus")
    role_id: Optional[str] = Field(None, validation_alias="RoleId")
    role_code: Optional[str] = Field(None, validation_alias="RoleCode")
    workspaces: Optional[Any] = Field(None, validation_alias="Workspaces")


class TokenPlanAccount(_Response):
    account_id: Optional[str] = Field(None, validation_alias="AccountId")
    account_type: Optional[str] = Field(None, validation_alias="AccountType")
    is_deleted: Optional[bool] = Field(None, validation_alias="IsDeleted")
    name: Optional[str] = Field(None, validation_alias="Name")
    aliyun_uid: Optional[str] = Field(None, validation_alias="AliyunUid")
    email: Optional[str] = Field(None, validation_alias="Email")
    created_at: Optional[str] = Field(None, validation_alias="CreatedAt")
    org_memberships: List[OrgMembership] = Field(
        default_factory=list,
        validation_alias="OrgMemberships",
    )


class OrgMember(_Model):
    account_id: Optional[str] = Field(None, validation_alias="AccountId")
    account_name: Optional[str] = Field(None, validation_alias="AccountName")
    account_biz_id: Optional[str] = Field(
        None,
        validation_alias="AccountBizId",
    )
    email: Optional[str] = Field(None, validation_alias="Email")
    org_id: Optional[str] = Field(None, validation_alias="OrgId")
    status: Optional[str] = Field(None, validation_alias="Status")
    roles: List[str] = Field(default_factory=list, validation_alias="Roles")
    gmt_create: Optional[Any] = Field(None, validation_alias="GmtCreate")
    seat_id: Optional[str] = Field(None, validation_alias="SeatId")
    spec_type: Optional[str] = Field(None, validation_alias="SpecType")
    api_key_id: Optional[Any] = Field(None, validation_alias="ApiKeyId")
    masked_api_key: Optional[str] = Field(
        None,
        validation_alias="MaskedApiKey",
    )
    pack_limit_info: Optional[Any] = Field(
        None,
        validation_alias="PackLimitInfo",
    )
    subscription_info: Optional[Any] = Field(
        None,
        validation_alias="SubscriptionInfo",
    )


class Seat(_Model):
    instance_code: Optional[str] = Field(None, validation_alias="InstanceCode")
    equity_list: List[Any] = Field(
        default_factory=list,
        validation_alias="EquityList",
    )
    end_time: Optional[Any] = Field(None, validation_alias="EndTime")
    seat_id: Optional[str] = Field(None, validation_alias="SeatId")
    spec_type: Optional[str] = Field(None, validation_alias="SpecType")
    start_time: Optional[Any] = Field(None, validation_alias="StartTime")
    assigned_status: Optional[str] = Field(
        None,
        validation_alias="AssignedStatus",
    )
    account_id: Optional[str] = Field(None, validation_alias="AccountId")
    account_name: Optional[str] = Field(None, validation_alias="AccountName")
    account_email: Optional[str] = Field(None, validation_alias="AccountEmail")
    status: Optional[str] = Field(None, validation_alias="Status")


class SeatStats(_Response):
    org_id: Optional[str] = Field(None, validation_alias="OrgId")
    total_member_count: Optional[int] = Field(
        None,
        validation_alias="TotalMemberCount",
    )
    owner_role_user_count: Optional[int] = Field(
        None,
        validation_alias="OwnerRoleUserCount",
    )
    admin_role_user_count: Optional[int] = Field(
        None,
        validation_alias="AdminRoleUserCount",
    )
    member_role_user_count: Optional[int] = Field(
        None,
        validation_alias="MemberRoleUserCount",
    )
    seated_member_count: Optional[int] = Field(
        None,
        validation_alias="SeatedMemberCount",
    )
    unseated_member_count: Optional[int] = Field(
        None,
        validation_alias="UnseatedMemberCount",
    )


class SubscriptionSeatItem(_Model):
    seat_type: Optional[str] = Field(None, validation_alias="SeatType")
    seat_refresh_time: Optional[int] = Field(
        None,
        validation_alias="SeatRefreshTime",
    )
    total_seats: Optional[int] = Field(None, validation_alias="TotalSeats")
    assigned_seats: Optional[int] = Field(
        None,
        validation_alias="AssignedSeats",
    )
    seat_credits: Optional[Any] = Field(None, validation_alias="SeatCredits")
    seat_remaining_credits: Optional[Any] = Field(
        None,
        validation_alias="SeatRemainingCredits",
    )


class SubscriptionStats(_Response):
    subscription_start_time: Optional[int] = Field(
        None,
        validation_alias="SubscriptionStartTime",
    )
    subscription_end_time: Optional[int] = Field(
        None,
        validation_alias="SubscriptionEndTime",
    )
    items: List[SubscriptionSeatItem] = Field(
        default_factory=list,
        validation_alias="Items",
    )


class SharedPackage(_Model):
    instance_code: Optional[str] = Field(None, validation_alias="InstanceCode")
    equity_list: List[Any] = Field(
        default_factory=list,
        validation_alias="EquityList",
    )
    status: Optional[str] = Field(None, validation_alias="Status")


# --- offset pages (tokenplan) --------------------------------------------


class OrgMemberPage(BasePage[OrgMember]):
    members: List[OrgMember] = Field(default_factory=list)

    @property
    def items(self) -> List[OrgMember]:
        return self.members


class SeatPage(BasePage[Seat]):
    seats: List[Seat] = Field(default_factory=list)

    @property
    def items(self) -> List[Seat]:
        return self.seats


class SharedPackagePage(BasePage[SharedPackage]):
    packages: List[SharedPackage] = Field(default_factory=list)

    @property
    def items(self) -> List[SharedPackage]:
        return self.packages
