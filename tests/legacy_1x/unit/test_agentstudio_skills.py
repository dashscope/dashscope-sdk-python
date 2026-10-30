# -*- coding: utf-8 -*-
"""Skill resource shape & auto-upload compatibility.

Tests create a client with a mocked transport and patch Files.upload
on the client's files instance since the SDK uses client instance mode.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dashscope.agentstudio import Client
from dashscope.agentstudio.transport import APIResponse
from dashscope.agentstudio.resources.skills import SkillVersions


class _Tx:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def request(self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        if "/versions" in path:
            return APIResponse(
                data={"id": "skv_1", "skill_id": "skl_1", "version": 1},
                request_id="req_2",
            )
        if path == "/skills":
            return APIResponse(
                data={"id": "skl_1", "name": "demo"},
                request_id="req_1",
            )
        return APIResponse(data={}, request_id=None)


@pytest.fixture(name="client")
def _client_fixture():
    """Create a client with a recording transport."""
    c = Client(api_key="test-key", base_url="http://test")
    c.transport = _Tx()
    return c


@pytest.fixture(name="fake_upload")
def _fake_upload_fixture(client, monkeypatch):
    """Patch Files.upload on the client's files instance to record calls."""
    calls: List[Dict[str, Any]] = []
    file_id_holder = {"id": "file_auto"}

    def _upload(_self_files, file, *, mime_type=None, **_kwargs):
        calls.append({"file": file, "mime_type": mime_type})

        class _F:
            id = file_id_holder["id"]

        return _F()

    monkeypatch.setattr(type(client.files), "upload", _upload)
    return calls, file_id_holder


def test_skill_create_with_file_id_does_not_call_upload(client, fake_upload):
    upload_calls, _ = fake_upload
    client.skills.create(file_id="file_existing")

    assert upload_calls == []
    body = client.transport.calls[0]["json"]
    assert body == {"file_id": "file_existing"}


def test_skill_create_with_file_path_auto_uploads_first(client, fake_upload):
    upload_calls, holder = fake_upload
    holder["id"] = "file_uploaded"

    client.skills.create(file="/tmp/skill.zip")

    assert len(upload_calls) == 1
    assert upload_calls[0]["file"] == "/tmp/skill.zip"
    assert upload_calls[0]["mime_type"] == "application/zip"
    body = client.transport.calls[0]["json"]
    assert body["file_id"] == "file_uploaded"


def test_skill_create_rejects_both_file_and_file_id(client):
    with pytest.raises(TypeError):
        client.skills.create(file_id="x", file="/tmp/y.zip")


def test_skill_create_rejects_neither_file_nor_file_id(client):
    with pytest.raises(TypeError):
        client.skills.create()


def test_skill_versions_create_auto_upload(client, fake_upload):
    upload_calls, holder = fake_upload
    holder["id"] = "file_v2"

    sv = SkillVersions(client)
    sv.create("skl_1", file="/tmp/v2.zip")

    assert len(upload_calls) == 1
    assert upload_calls[0]["file"] == "/tmp/v2.zip"
    body = client.transport.calls[0]["json"]
    assert body == {"file_id": "file_v2"}


def test_skill_versions_create_with_file_id_no_upload(client, fake_upload):
    upload_calls, _ = fake_upload

    sv = SkillVersions(client)
    sv.create("skl_1", file_id="file_existing")

    assert upload_calls == []
    body = client.transport.calls[0]["json"]
    assert body == {"file_id": "file_existing"}


def test_skill_versions_create_explicit_mime_type_overrides(
    client,
    fake_upload,
):
    upload_calls, _ = fake_upload

    sv = SkillVersions(client)
    sv.create("skl_1", file="/tmp/v.tar.gz", mime_type="application/gzip")
    assert upload_calls[0]["mime_type"] == "application/gzip"
