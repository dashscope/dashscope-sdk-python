# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared helper functions for resource modules."""

from __future__ import annotations

from typing import Any, Mapping

from dashscope.agentstudio.types import (
    Agent,
    AgentVersion,
    Credential,
    Deployment,
    DeploymentRun,
    Environment,
    File,
    ServerEvent,
    Session,
    SessionResource,
    SessionThread,
    Skill,
    SkillVersion,
    Vault,
    WebhookEndpoint,
    WebhookEvent,
    WebhookSecretReset,
    parse_server_event,
)


# ---------------------------------------------------------------------------
# Coerce helpers
# ---------------------------------------------------------------------------


def _coerce_agent(payload: Mapping[str, Any]) -> Agent:
    return Agent(**dict(payload))


def _coerce_agent_version(payload: Mapping[str, Any]) -> AgentVersion:
    return AgentVersion(**dict(payload))


def _coerce_env(payload: Mapping[str, Any]) -> Environment:
    return Environment(**dict(payload))


def _coerce_file(payload: Mapping[str, Any]) -> File:
    return File(**dict(payload))


def _coerce_skill(payload: Mapping[str, Any]) -> Skill:
    return Skill(**dict(payload))


def _coerce_skill_version(payload: Mapping[str, Any]) -> SkillVersion:
    return SkillVersion(**dict(payload))


def _coerce_event(payload: Mapping[str, Any]) -> ServerEvent:
    return parse_server_event(dict(payload))


def _coerce_session(payload: Mapping[str, Any]) -> Session:
    return Session(**dict(payload))


def _coerce_session_resource(payload: Mapping[str, Any]) -> SessionResource:
    return SessionResource(**dict(payload))


def _coerce_session_thread(payload: Mapping[str, Any]) -> SessionThread:
    return SessionThread(**dict(payload))


def _coerce_vault(payload: Mapping[str, Any]) -> Vault:
    return Vault(**dict(payload))


def _coerce_credential(payload: Mapping[str, Any]) -> Credential:
    return Credential(**dict(payload))


def _coerce_webhook_endpoint(payload: Mapping[str, Any]) -> WebhookEndpoint:
    return WebhookEndpoint(**dict(payload))


def _coerce_webhook_event(payload: Mapping[str, Any]) -> WebhookEvent:
    return WebhookEvent(**dict(payload))


def _coerce_webhook_secret_reset(
    payload: Mapping[str, Any],
) -> WebhookSecretReset:
    return WebhookSecretReset(**dict(payload))


def _coerce_deployment(payload: Mapping[str, Any]) -> Deployment:
    return Deployment(**dict(payload))


def _coerce_deployment_run(payload: Mapping[str, Any]) -> DeploymentRun:
    return DeploymentRun(**dict(payload))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _events_path(session_id: str) -> str:
    return f"/sessions/{session_id}/events"


def _stream_path(session_id: str) -> str:
    return f"/sessions/{session_id}/events/stream"


def _session_resources_path(session_id: str) -> str:
    return f"/sessions/{session_id}/resources"


def _session_resource_item_path(session_id: str, resource_id: str) -> str:
    return f"/sessions/{session_id}/resources/{resource_id}"


def _session_threads_path(session_id: str) -> str:
    return f"/sessions/{session_id}/threads"


def _session_thread_item_path(session_id: str, thread_id: str) -> str:
    return f"/sessions/{session_id}/threads/{thread_id}"


def _session_thread_events_path(session_id: str, thread_id: str) -> str:
    return f"/sessions/{session_id}/threads/{thread_id}/events"


def _session_thread_archive_path(session_id: str, thread_id: str) -> str:
    return f"/sessions/{session_id}/threads/{thread_id}/archive"
