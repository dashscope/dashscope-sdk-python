# -*- coding: utf-8 -*-
"""Unit tests for Vault, Credential, DeleteResponse, Agent.system_prompt."""

import pytest

from dashscope.agentstudio.resources._helpers import (
    _coerce_vault,
    _coerce_credential,
)
from dashscope.agentstudio.types import (
    Agent,
    Credential,
    CredentialAuth,
    DeleteResponse,
    Environment,
    File,
    ModelConfig,
    Networking,
    Vault,
)


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------


def test_vault_coerce_parses_all_fields():
    payload = {
        "id": "vlt_1",
        "type": "vault",
        "display_name": "my-vault",
        "metadata": {"team": "backend"},
        "created_at": "2026-06-29T10:00:00Z",
        "updated_at": "2026-06-29T10:00:00Z",
        "archived_at": None,
        "request_id": "req_1",
    }
    v = _coerce_vault(payload)
    assert isinstance(v, Vault)
    assert v.id == "vlt_1"
    assert v.display_name == "my-vault"
    assert v.metadata == {"team": "backend"}
    assert v.archived_at is None


def test_vault_skips_none_fields_in_repr():
    v = Vault(id="vlt_1", display_name="demo")
    r = repr(v)
    assert "id='vlt_1'" in r
    assert "display_name='demo'" in r
    assert "archived_at" not in r  # None fields omitted


# ---------------------------------------------------------------------------
# CredentialAuth — three auth types
# ---------------------------------------------------------------------------


def test_credential_auth_environment_variable():
    auth = CredentialAuth(
        type="environment_variable",
        secret_name="OPENAI_API_KEY",
        secret_value="sk-xxx",
        networking={"type": "unrestricted"},
    )
    assert auth.type == "environment_variable"
    assert auth.secret_name == "OPENAI_API_KEY"
    assert isinstance(auth.networking, Networking)
    assert auth.networking.type == "unrestricted"


def test_credential_auth_static_bearer():
    auth = CredentialAuth(
        type="static_bearer",
        token="Bearer eyJhbGci...",
    )
    assert auth.type == "static_bearer"
    assert auth.token == "Bearer eyJhbGci..."
    assert auth.networking is None


def test_credential_auth_mcp_oauth():
    auth = CredentialAuth(
        type="mcp_oauth",
        mcp_server_url="https://mcp.example.com/oauth",
    )
    assert auth.type == "mcp_oauth"
    assert auth.mcp_server_url == "https://mcp.example.com/oauth"


def test_credential_auth_secret_value_not_returned_by_server():
    """Server stores secret_value write-only."""
    auth = CredentialAuth(
        type="environment_variable",
        secret_name="KEY",
        # secret_value intentionally absent (server doesn't return it)
    )
    assert auth.secret_value is None


# ---------------------------------------------------------------------------
# Credential — nested auth parsing
# ---------------------------------------------------------------------------


def test_credential_coerce_wraps_auth_dict():
    payload = {
        "id": "vcrd_1",
        "type": "vault_credential",
        "vault_id": "vlt_1",
        "display_name": "OpenAI Key",
        "auth": {
            "type": "environment_variable",
            "secret_name": "OPENAI_API_KEY",
            "networking": {"type": "restricted"},
        },
        "created_at": "2026-06-29T10:00:00Z",
        "updated_at": "2026-06-29T10:00:00Z",
        "metadata": {"service": "openai"},
        "archived_at": None,
        "request_id": "req_1",
    }
    c = _coerce_credential(payload)
    assert isinstance(c, Credential)
    assert c.id == "vcrd_1"
    assert c.vault_id == "vlt_1"
    assert isinstance(c.auth, CredentialAuth)
    assert c.auth.type == "environment_variable"
    assert isinstance(c.auth.networking, Networking)
    assert c.auth.networking.type == "restricted"


def test_credential_coerce_static_bearer():
    payload = {
        "id": "vcrd_2",
        "type": "vault_credential",
        "vault_id": "vlt_1",
        "auth": {"type": "static_bearer", "token": "Bearer xyz"},
        "display_name": "API Token",
    }
    c = _coerce_credential(payload)
    assert c.auth.type == "static_bearer"
    assert c.auth.token == "Bearer xyz"


# ---------------------------------------------------------------------------
# DeleteResponse
# ---------------------------------------------------------------------------


def test_delete_response_fields():
    dr = DeleteResponse(id="vlt_1", type="vault_deleted", request_id="req_1")
    assert dr.id == "vlt_1"
    assert dr.type == "vault_deleted"
    assert dr.request_id == "req_1"


def test_delete_response_repr_skips_none():
    dr = DeleteResponse(id="x", type="deleted")
    assert "request_id" not in repr(dr)


# ---------------------------------------------------------------------------
# Agent.system_prompt property alias (gotcha: create param is system_prompt,
# field is system, property bridges them)
# ---------------------------------------------------------------------------


def test_agent_system_prompt_is_alias_for_system():
    agent = Agent(
        id="agent_1",
        name="demo",
        model={"id": "qwen-plus"},
        system="你是一个助手。",
    )
    assert agent.system == "你是一个助手。"
    assert agent.system_prompt == agent.system


def test_agent_system_prompt_none_when_system_unset():
    agent = Agent(id="agent_1", name="demo", model={"id": "qwen-plus"})
    assert agent.system is None
    assert agent.system_prompt is None


# ---------------------------------------------------------------------------
# Agent model config (model.id + inference params)
# ---------------------------------------------------------------------------


_FULL_MODEL = {
    "id": "qwen3-max",
    "name": "Qwen3 Max",
    "temperature": 0.7,
    "max_tokens": 8192,
    "enable_thinking": True,
    "thinking_budget": 2048,
}


def test_agent_model_hydrates_into_model_config():
    """The ``model`` object is typed, so inference params are attribute-readable."""
    agent = Agent(id="agent_1", model=dict(_FULL_MODEL))
    assert isinstance(agent.model, ModelConfig)
    assert agent.model.id == "qwen3-max"
    assert agent.model.name == "Qwen3 Max"
    assert agent.model.temperature == 0.7
    assert agent.model.max_tokens == 8192
    assert agent.model.enable_thinking is True
    assert agent.model.thinking_budget == 2048
    assert agent.model_id == "qwen3-max"


def test_agent_model_reads_typed_and_serializes_via_to_dict():
    """Attributes carry wire names; ``to_dict()`` is the dict view."""
    agent = Agent(id="agent_1", model=dict(_FULL_MODEL))
    assert agent.model.to_dict() == _FULL_MODEL
    assert agent.model.max_tokens == 8192
    with pytest.raises(TypeError):
        _ = agent.model["id"]


def test_agent_model_omits_unset_params_and_keeps_unknown_fields():
    """Unset fields omitted; unknown fields preserved."""
    agent = Agent(id="agent_1", model={"id": "qwen3-max", "vendor": "qwen"})
    assert agent.model.to_dict() == {"id": "qwen3-max", "vendor": "qwen"}
    assert agent.model.temperature is None
    assert agent.model_id == "qwen3-max"


def test_agent_model_absent_stays_none():
    agent = Agent(id="agent_1", name="demo")
    assert agent.model is None
    assert agent.model_id is None


def test_agent_model_accepts_model_config_on_construction():
    """A ModelConfig round-trips back into an Agent."""
    config = ModelConfig(**_FULL_MODEL)
    agent = Agent(id="agent_1", model=config)
    assert agent.model is config
    assert Agent(**agent.to_dict()).model.temperature == 0.7


def test_model_config_copy_with_keeps_the_rest():
    """copy_with() is the read-modify-write helper for update."""
    config = ModelConfig(**_FULL_MODEL)
    tuned = config.copy_with(temperature=0.1)
    assert tuned.temperature == 0.1
    assert tuned.to_dict() == dict(_FULL_MODEL, temperature=0.1)
    assert config.temperature == 0.7  # original untouched
    assert isinstance(tuned, ModelConfig)


def test_model_config_copy_with_can_drop_and_reject_typos():
    config = ModelConfig(**_FULL_MODEL)
    # None drops the field: "let the server apply its default"
    assert (
        "thinking_budget"
        not in config.copy_with(
            thinking_budget=None,
        ).to_dict()
    )
    # typos fail loudly instead of being sent as unknown wire fields
    with pytest.raises(ValueError, match="unknown model config field"):
        config.copy_with(temp=0.5)


def test_model_config_unknown_fields_survive_copy_with():
    """Forward-compat fields survive copy_with and can be retuned."""
    config = ModelConfig(id="qwen3-max", vendor="qwen")
    assert config.copy_with(temperature=0.5).to_dict() == {
        "id": "qwen3-max",
        "vendor": "qwen",
        "temperature": 0.5,
    }
    assert config.copy_with(vendor="other").to_dict() == {
        "id": "qwen3-max",
        "vendor": "other",
    }
    # still not a way to invent wire fields: neither known nor carried
    with pytest.raises(ValueError, match="unknown model config field"):
        config.copy_with(vendorx="nope")


# ---------------------------------------------------------------------------
# Environment config nested parsing
# ---------------------------------------------------------------------------


def test_environment_config_nested_objects():
    env = Environment(
        id="env_1",
        name="python-env",
        config={
            "type": "cloud",
            "networking": {"type": "unrestricted"},
            "packages": {"pip": ["pandas", "numpy"]},
        },
    )
    assert env.config.type == "cloud"
    assert isinstance(env.config.networking, Networking)
    assert env.config.networking.type == "unrestricted"
    assert env.config.packages.pip == ["pandas", "numpy"]


# ---------------------------------------------------------------------------
# File model
# ---------------------------------------------------------------------------


def test_file_model_fields():
    f = File(
        id="file_1",
        filename="doc.pdf",
        size_bytes=1024,
        mime_type="application/pdf",
        status="checking",
    )
    assert f.id == "file_1"
    assert f.filename == "doc.pdf"
    assert f.size_bytes == 1024
    assert f.status == "checking"
    assert f.downloadable is None  # not set
