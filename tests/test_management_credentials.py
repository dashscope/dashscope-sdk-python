# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Credential-chain resolution tests for the management surface."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.management import _credentials  # noqa: E402
from dashscope.management._credentials import (  # noqa: E402
    CredentialResolver,
    ResolvedCredential,
)

AK = "ak-test"
SK = "sk-test"
TOKEN = "sts-token-test"


class _FakeCredentialClient:
    """Stands in for a pre-built alibabacloud credentials Client."""

    def __init__(self, ak="fake-ak", sk="fake-sk", token=None, provider=None):
        self._model = SimpleNamespace(
            access_key_id=ak,
            access_key_secret=sk,
            security_token=token,
            provider_name=provider,
        )

    def get_credential(self):
        return self._model

    async def get_credential_async(self):
        return self._model


def test_access_key_resolution():
    resolver = CredentialResolver(access_key_id=AK, access_key_secret=SK)
    cred = resolver.resolve()
    assert isinstance(cred, ResolvedCredential)
    assert cred.access_key_id == AK
    assert cred.access_key_secret == SK
    assert cred.security_token is None


def test_sts_resolution_carries_token():
    resolver = CredentialResolver(
        access_key_id=AK,
        access_key_secret=SK,
        security_token=TOKEN,
    )
    cred = resolver.resolve()
    assert cred.security_token == TOKEN


def test_config_dict_resolution():
    resolver = CredentialResolver(
        config={
            "type": "access_key",
            "access_key_id": AK,
            "access_key_secret": SK,
        },
    )
    cred = resolver.resolve()
    assert cred.access_key_id == AK
    assert cred.access_key_secret == SK


def test_prebuilt_credential_wins_over_ak():
    fake = _FakeCredentialClient(ak="prebuilt-ak", sk="prebuilt-sk")
    resolver = CredentialResolver(
        credential=fake,
        access_key_id="ignored-ak",
        access_key_secret="ignored-sk",
    )
    cred = resolver.resolve()
    assert cred.access_key_id == "prebuilt-ak"


def test_config_wins_over_ak():
    resolver = CredentialResolver(
        config={
            "type": "access_key",
            "access_key_id": "config-ak",
            "access_key_secret": "config-sk",
        },
        access_key_id="ignored-ak",
        access_key_secret="ignored-sk",
    )
    assert resolver.resolve().access_key_id == "config-ak"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"access_key_id": AK},
        {"access_key_secret": SK},
        {"security_token": TOKEN},
        {"access_key_id": AK, "security_token": TOKEN},
        {"access_key_id": AK, "access_key_secret": ""},
    ],
)
def test_partial_explicit_credential_rejected(kwargs):
    # A partial AK must fail loudly, never sign with ambient credentials.
    with pytest.raises(ValueError, match="together"):
        CredentialResolver(**kwargs)


def test_default_chain_only_without_explicit_credential(monkeypatch):
    seen = []

    class _StubClient:
        def __init__(self, config=None):
            seen.append(config)

    monkeypatch.setattr(_credentials, "_CredentialsClient", _StubClient)
    CredentialResolver()
    assert seen == [None]


def test_provider_name_propagates():
    fake = _FakeCredentialClient(provider="ecs_ram_role")
    cred = CredentialResolver(credential=fake).resolve()
    assert cred.provider_name == "ecs_ram_role"


async def test_resolve_async_matches_sync():
    resolver = CredentialResolver(access_key_id=AK, access_key_secret=SK)
    sync_cred = resolver.resolve()
    async_cred = await resolver.resolve_async()
    assert async_cred == sync_cred


async def test_prebuilt_credential_async():
    fake = _FakeCredentialClient(ak="async-ak", token=TOKEN)
    cred = await CredentialResolver(credential=fake).resolve_async()
    assert cred.access_key_id == "async-ak"
    assert cred.security_token == TOKEN


def test_secret_never_logged(caplog):
    resolver = CredentialResolver(access_key_id=AK, access_key_secret=SK)
    with caplog.at_level(logging.DEBUG):
        resolver.resolve()
    assert SK not in caplog.text
    assert AK not in caplog.text


def test_snapshot_is_frozen():
    cred = CredentialResolver(
        access_key_id=AK,
        access_key_secret=SK,
    ).resolve()
    with pytest.raises(Exception):
        cred.access_key_id = "mutated"
