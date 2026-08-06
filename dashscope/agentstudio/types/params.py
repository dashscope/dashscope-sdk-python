# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-resource API request parameter definitions.

Each resource group (Agents, Environments, Sessions, …) defines the
request-body structures for its write endpoints.  All classes inherit
from :class:`~dashscope.agentstudio.types.models.BaseModel` so that
``.to_dict()`` produces a ready-to-send wire-format dict.

The ``__init__`` signature uses **user-friendly** parameter names
(e.g. ``system_prompt``) while the ``super().__init__()`` call passes
**wire-format** field names (e.g. ``system``).

Python 3.8+ compatible.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashscope.agentstudio.types.models import BaseModel

_NOT_GIVEN = object()


def _to_mapping(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return value


# ===========================================================================
# Agents
# ===========================================================================


class AgentCreateParams(BaseModel):
    """Request body for ``POST /agents``.

    Maps to :meth:`Agents.create`.  The SDK method exposes
    ``model`` as a plain string and ``system_prompt`` as a keyword arg,
    but the wire format uses ``model: {"id": "…"}`` and ``system``.
    """

    _fields = (
        "name",
        "model",
        "description",
        "system",
        "tools",
        "mcp_servers",
        "skills",
        "metadata",
    )

    def __init__(
        self,
        *,
        name: str,
        model: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        mcp_servers: Optional[Sequence[Mapping[str, Any]]] = None,
        skills: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            name=name,
            model={"id": model},
            description=description,
            system=system_prompt,
            tools=([dict(t) for t in tools] if tools is not None else None),
            mcp_servers=(
                [dict(s) for s in mcp_servers]
                if mcp_servers is not None
                else None
            ),
            skills=([dict(s) for s in skills] if skills is not None else None),
            metadata=(dict(metadata) if metadata is not None else None),
        )


class AgentUpdateParams(BaseModel):
    """Request body for ``POST /agents/{id}`` (update latest version)."""

    _fields = (
        "version",
        "name",
        "model",
        "description",
        "system",
        "tools",
        "mcp_servers",
        "skills",
        "metadata",
    )

    def __init__(
        self,
        *,
        version: int,
        name: Optional[str] = None,
        model: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        mcp_servers: Optional[Sequence[Mapping[str, Any]]] = None,
        skills: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            version=version,
            name=name,
            model=({"id": model} if model is not None else None),
            description=description,
            system=system_prompt,
            tools=([dict(t) for t in tools] if tools is not None else None),
            mcp_servers=(
                [dict(s) for s in mcp_servers]
                if mcp_servers is not None
                else None
            ),
            skills=([dict(s) for s in skills] if skills is not None else None),
            metadata=(dict(metadata) if metadata is not None else None),
        )


class AgentListParams(BaseModel):
    """Query params for ``GET /agents``.

    ``include_archived`` is a Python bool mapped to the wire-format
    string ``"true"`` / ``"false"``.
    """

    _fields = ("limit", "page", "include_archived")

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> None:
        archived = (
            str(include_archived).lower()
            if include_archived is not None
            else None
        )
        super().__init__(
            limit=limit,
            page=page,
            include_archived=archived,
        )


class AgentVersionListParams(BaseModel):
    """Query params for ``GET /agents/{id}/versions``."""

    _fields = ("limit", "page")


# ===========================================================================
# Environments
# ===========================================================================


class EnvironmentCreateParams(BaseModel):
    """Request body for ``POST /environments``."""

    _fields = ("name", "config", "description", "scope", "metadata")

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        name: str,
        config: Mapping[str, Any],
        description: Optional[str] = None,
        scope: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            name=name,
            config=dict(config),
            description=description,
            scope=scope,
            metadata=(dict(metadata) if metadata is not None else None),
        )


class EnvironmentUpdateParams(BaseModel):
    """Request body for ``POST /environments/{id}``."""

    _fields = ("name", "description", "config", "scope", "metadata")

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        scope: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            config=(dict(config) if config is not None else None),
            scope=scope,
            metadata=(dict(metadata) if metadata is not None else None),
        )


class EnvironmentListParams(BaseModel):
    """Query params for ``GET /environments``.

    ``include_archived`` is a Python bool mapped to the wire-format
    string ``"true"`` / ``"false"``.
    """

    _fields = ("limit", "page", "include_archived")

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> None:
        archived = (
            str(include_archived).lower()
            if include_archived is not None
            else None
        )
        super().__init__(
            limit=limit,
            page=page,
            include_archived=archived,
        )


# ===========================================================================
# Sessions
# ===========================================================================


class SessionCreateParams(BaseModel):
    """Request body for ``POST /sessions``.

    ``agent`` is the agent ID string (not the full agent object).
    ``resources`` is an optional list of file mounts; each item is a
    mapping with ``type``, ``file_id`` and ``mount_path`` keys.
    """

    _fields = ("agent", "environment_id", "title", "resources", "metadata")

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        agent: str,
        environment_id: Optional[str] = None,
        title: Optional[str] = None,
        resources: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent=agent,
            environment_id=environment_id,
            title=title,
            resources=(
                [dict(r) for r in resources] if resources is not None else None
            ),
            metadata=(dict(metadata) if metadata is not None else None),
        )


class SessionUpdateParams(BaseModel):
    """Request body for ``POST /sessions/{id}``."""

    _fields = ("title", "metadata")

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        title: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            title=title,
            metadata=(dict(metadata) if metadata is not None else None),
        )


class SessionListParams(BaseModel):
    """Query params for ``GET /sessions``.

    ``statuses`` is a list of session status strings such as
    ``"idle"``, ``"running"``, ``"terminated"``.

    Python-friendly parameter names (``statuses``, ``created_at_gt``, …)
    are mapped to wire-format keys (``statuses[]``, ``created_at[gt]``, …)
    inside ``__init__``.
    """

    _fields = (
        "limit",
        "page",
        "agent_id",
        "statuses[]",
        "created_at[gt]",
        "created_at[gte]",
        "created_at[lt]",
        "created_at[lte]",
    )

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        agent_id: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        created_at_gt: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lt: Optional[str] = None,
        created_at_lte: Optional[str] = None,
    ) -> None:
        kwargs = {}
        kwargs["limit"] = limit
        kwargs["page"] = page
        kwargs["agent_id"] = agent_id
        if statuses is not None:
            kwargs["statuses[]"] = list(statuses)
        if created_at_gt is not None:
            kwargs["created_at[gt]"] = created_at_gt
        if created_at_gte is not None:
            kwargs["created_at[gte]"] = created_at_gte
        if created_at_lt is not None:
            kwargs["created_at[lt]"] = created_at_lt
        if created_at_lte is not None:
            kwargs["created_at[lte]"] = created_at_lte
        super().__init__(**kwargs)


# ===========================================================================
# SessionEvents
# ===========================================================================


class SessionEventSendParams(BaseModel):
    """Request body for ``POST /sessions/{id}/events``.

    The ``input`` array must contain at least one event.
    """

    _fields = ("input",)

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        input: Sequence[  # pylint: disable=redefined-builtin
            Mapping[str, Any]
        ],
    ) -> None:
        super().__init__(
            input=[dict(e) for e in input],
        )


class SessionEventListParams(BaseModel):
    """Query params for ``GET /sessions/{id}/events``.

    ``types`` is a list of event type strings joined by comma in the
    wire format.  ``created_at_*`` parameters are mapped to the
    bracket-based wire keys ``created_at[gt]`` etc.
    """

    _fields = (
        "types",
        "created_at[gt]",
        "created_at[gte]",
        "created_at[lt]",
        "created_at[lte]",
        "limit",
        "order",
        "page",
    )

    def __init__(
        self,
        *,
        types: Optional[Sequence[str]] = None,
        created_at_gt: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lt: Optional[str] = None,
        created_at_lte: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        page: Optional[str] = None,
    ) -> None:
        kwargs = {}
        if types is not None:
            kwargs["types"] = ",".join(types)
        if created_at_gt is not None:
            kwargs["created_at[gt]"] = created_at_gt
        if created_at_gte is not None:
            kwargs["created_at[gte]"] = created_at_gte
        if created_at_lt is not None:
            kwargs["created_at[lt]"] = created_at_lt
        if created_at_lte is not None:
            kwargs["created_at[lte]"] = created_at_lte
        kwargs["limit"] = limit
        kwargs["order"] = order
        kwargs["page"] = page
        super().__init__(**kwargs)


# ===========================================================================
# Deployments
# ===========================================================================


class DeploymentCreateParams(BaseModel):
    """Request body for ``POST /deployments``."""

    _fields = (
        "name",
        "description",
        "agent",
        "environment_id",
        "schedule",
        "initial_events",
        "resources",
        "vault_ids",
        "metadata",
    )

    def __init__(
        self,
        *,
        name: str,
        agent: Any,
        initial_events: Sequence[Mapping[str, Any]],
        description: Optional[str] = None,
        environment_id: Optional[str] = None,
        schedule: Any = None,
        resources: Optional[Sequence[Any]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        BaseModel.__init__(
            self,
            name=name,
            description=description,
            agent=_to_mapping(agent),
            environment_id=environment_id,
            schedule=(_to_mapping(schedule) if schedule is not None else None),
            initial_events=[dict(event) for event in initial_events],
            resources=(
                [_to_mapping(resource) for resource in resources]
                if resources is not None
                else None
            ),
            vault_ids=(list(vault_ids) if vault_ids is not None else None),
            metadata=(dict(metadata) if metadata is not None else None),
        )


class DeploymentUpdateParams(BaseModel):
    """Request body for ``POST /deployments/{deployment_id}``.

    ``environment_id`` and ``schedule`` preserve the distinction between an
    omitted argument and an explicit ``None``. Explicit ``None`` is emitted as
    JSON ``null`` to clear the field.
    """

    _fields = (
        "name",
        "description",
        "agent",
        "environment_id",
        "schedule",
        "initial_events",
        "resources",
        "vault_ids",
        "metadata",
    )

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agent: Any = None,
        environment_id: Any = _NOT_GIVEN,
        schedule: Any = _NOT_GIVEN,
        initial_events: Optional[Sequence[Mapping[str, Any]]] = None,
        resources: Optional[Sequence[Any]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._environment_id_given = environment_id is not _NOT_GIVEN
        self._schedule_given = schedule is not _NOT_GIVEN
        super().__init__(
            name=name,
            description=description,
            agent=(_to_mapping(agent) if agent is not None else None),
            environment_id=(
                environment_id if self._environment_id_given else None
            ),
            schedule=(
                _to_mapping(schedule)
                if self._schedule_given and schedule is not None
                else None
            ),
            initial_events=(
                [dict(event) for event in initial_events]
                if initial_events is not None
                else None
            ),
            resources=(
                [_to_mapping(resource) for resource in resources]
                if resources is not None
                else None
            ),
            vault_ids=(list(vault_ids) if vault_ids is not None else None),
            metadata=(dict(metadata) if metadata is not None else None),
        )

    def to_dict(self) -> dict:
        body = super().to_dict()
        if self._environment_id_given and self.environment_id is None:
            body["environment_id"] = None
        if self._schedule_given and self.schedule is None:
            body["schedule"] = None
        return body


class DeploymentListParams(BaseModel):
    """Query params for ``GET /deployments``."""

    _fields = (
        "agent_id",
        "keyword",
        "status",
        "include_archived",
        "limit",
        "page",
        "created_at[gte]",
        "created_at[lte]",
    )

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        keyword: Optional[str] = None,
        status: Optional[str] = None,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lte: Optional[str] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            keyword=keyword,
            status=status,
            include_archived=(
                str(include_archived).lower()
                if include_archived is not None
                else None
            ),
            limit=limit,
            page=page,
            **{
                "created_at[gte]": created_at_gte,
                "created_at[lte]": created_at_lte,
            },
        )


class DeploymentRunListParams(BaseModel):
    """Query params for deployment run list endpoints."""

    _fields = ("limit", "page")


# ===========================================================================
# Files
# ===========================================================================


class FileListParams(BaseModel):
    """Query params for ``GET /files``.

    File upload uses multipart form data and is therefore not
    represented as a param class.
    """

    _fields = ("limit", "page", "scope_id")


# ===========================================================================
# Skills
# ===========================================================================


class SkillCreateParams(BaseModel):
    """Request body for ``POST /skills``.

    ``file_id`` references a file previously uploaded via the Files API.
    The server extracts ``name`` and ``description`` from the zip's
    ``SKILL.md``; they are not client-settable.
    """

    _fields = ("file_id",)


class SkillListParams(BaseModel):
    """Query params for ``GET /skills``."""

    _fields = ("source", "limit", "page")


# ===========================================================================
# SkillVersions
# ===========================================================================


class SkillVersionCreateParams(BaseModel):
    """Request body for ``POST /skills/{skill_id}/versions``."""

    _fields = ("file_id",)


class SkillVersionListParams(BaseModel):
    """Query params for ``GET /skills/{skill_id}/versions``."""

    _fields = ("limit", "page")


# ===========================================================================
# Vaults
# ===========================================================================


class VaultCreateParams(BaseModel):
    """Request body for ``POST /vaults``."""

    _fields = ("display_name", "metadata")


class VaultUpdateParams(BaseModel):
    """Request body for ``POST /vaults/{vault_id}``."""

    _fields = ("display_name", "metadata")


class VaultListParams(BaseModel):
    """Query params for ``GET /vaults``."""

    _fields = ("include_archived", "limit", "page")

    def __init__(
        self,
        *,
        include_archived: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if include_archived is not None:
            kwargs["include_archived"] = str(
                include_archived,
            ).lower()
        super().__init__(**kwargs)


# ===========================================================================
# Credentials
# ===========================================================================


class CredentialCreateParams(BaseModel):
    """Request body for ``POST /vaults/{vault_id}/credentials``."""

    _fields = ("auth", "display_name", "metadata")

    def __init__(self, *, auth: Any = None, **kwargs: Any) -> None:
        if auth is not None:
            if hasattr(auth, "to_dict"):
                kwargs["auth"] = auth.to_dict()
            elif isinstance(auth, Mapping):
                kwargs["auth"] = dict(auth)
            else:
                kwargs["auth"] = auth
        super().__init__(**kwargs)


class CredentialUpdateParams(BaseModel):
    """Request body for ``POST /vaults/{v}/credentials/{c}``."""

    _fields = ("auth", "display_name", "metadata")

    def __init__(self, *, auth: Any = None, **kwargs: Any) -> None:
        if auth is not None:
            if hasattr(auth, "to_dict"):
                kwargs["auth"] = auth.to_dict()
            elif isinstance(auth, Mapping):
                kwargs["auth"] = dict(auth)
            else:
                kwargs["auth"] = auth
        super().__init__(**kwargs)


class CredentialListParams(BaseModel):
    """Query params for ``GET /vaults/{vault_id}/credentials``."""

    _fields = ("include_archived", "limit", "page")

    def __init__(
        self,
        *,
        include_archived: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if include_archived is not None:
            kwargs["include_archived"] = str(
                include_archived,
            ).lower()
        super().__init__(**kwargs)
