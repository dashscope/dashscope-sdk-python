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
    Skill,
    SkillVersion,
    Vault,
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


def _coerce_vault(payload: Mapping[str, Any]) -> Vault:
    return Vault(**dict(payload))


def _coerce_credential(payload: Mapping[str, Any]) -> Credential:
    return Credential(**dict(payload))


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
