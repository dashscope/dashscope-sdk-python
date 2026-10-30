# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import annotations

import importlib
import os
import pathlib
import subprocess
import sys
from unittest.mock import patch

import pytest

import dashscope
from dashscope.aigc.multimodal_conversation import MultiModalConversation
from dashscope.audio.qwen_asr import QwenTranscription
from dashscope.finetune.finetunes import FineTunes


_REPO = pathlib.Path(__file__).resolve().parents[1]


def _run_python(*args):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, *args],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_legacy_module_aliases_in_clean_process():
    _run_python(
        "-c",
        """
import importlib
import dashscope

packages = (
    'aigc', 'api_entities', 'app', 'assistants', 'audio', 'cli',
    'common', 'embeddings', 'io', 'multimodal', 'nlp', 'protocol',
    'rerank', 'threads', 'tokenizers', 'utils',
)
modules = ('files', 'model', 'models')
for name in packages + modules:
    old = importlib.import_module(f'dashscope.{name}')
    canonical = importlib.import_module(f'dashscope.legacy_1x.{name}')
    assert old is canonical, name
    assert getattr(dashscope, name) is old, name

for name in ('customize_types', 'deployments', 'finetunes'):
    old = importlib.import_module(f'dashscope.finetune.{name}')
    canonical = importlib.import_module(
        f'dashscope.legacy_1x.finetune.{name}',
    )
    assert old is canonical, name

# ``from dashscope.finetune import <sub>`` resolves the child against the
# shared canonical ``__name__``; guard the 1.x-only submodules stay reachable.
from dashscope.finetune import customize_types, deployments, finetunes
for name, mod in (
    ('customize_types', customize_types),
    ('deployments', deployments),
    ('finetunes', finetunes),
):
    legacy = importlib.import_module(f'dashscope.legacy_1x.finetune.{name}')
    assert mod is legacy, name
""",
    )


def test_target_legacy_classes_keep_root_identity():
    assert dashscope.FineTunes is FineTunes
    assert dashscope.MultiModalConversation is MultiModalConversation


@pytest.mark.parametrize(
    ("patch_target", "api"),
    [
        ("dashscope.finetune.finetunes.FineTunes.call", FineTunes),
        (
            "dashscope.aigc.multimodal_conversation.MultiModalConversation.call",
            MultiModalConversation,
        ),
        (
            "dashscope.audio.qwen_asr.qwen_transcription.QwenTranscription.call",
            QwenTranscription,
        ),
    ],
)
def test_target_legacy_calls_accept_old_patch_paths(patch_target, api):
    sentinel = object()
    with patch(patch_target, return_value=sentinel) as call:
        result = api.call(model="test-model")
    assert result is sentinel
    call.assert_called_once_with(model="test-model")


def test_client_and_legacy_base_api_coexist():
    from dashscope.client import AsyncClient, Client
    from dashscope.client.base_api import BaseApi

    canonical = importlib.import_module("dashscope.legacy_1x.client.base_api")
    assert importlib.import_module("dashscope.client.base_api") is canonical
    assert BaseApi is canonical.BaseApi
    assert Client is dashscope.Client
    assert AsyncClient is dashscope.AsyncClient
    assert Client.__module__ == "dashscope.client"


def test_leftover_1x_client_dir_does_not_shadow_client_module(tmp_path):
    """pip leaves 1.x files behind on an in-place upgrade, and a directory
    outranks a same-named module, so the empty 1.x ``client/`` would win."""
    site = tmp_path / "site-packages"
    installed = site / "dashscope"
    installed.mkdir(parents=True)
    for entry in (_REPO / "dashscope").iterdir():
        if entry.name == "__pycache__":
            continue
        (installed / entry.name).symlink_to(entry)

    stale = installed / "client"
    stale.mkdir()
    (stale / "__init__.py").write_text("", encoding="utf-8")
    (stale / "base_api.py").write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(site)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import dashscope
import dashscope.client as client
from dashscope.client import AsyncClient, Client
from dashscope.client.base_api import BaseApi

assert client.__file__.endswith('client.py'), client.__file__
assert Client is dashscope.Client
assert AsyncClient is dashscope.AsyncClient
assert Client.__module__ == 'dashscope.client'
assert BaseApi.__module__ == 'dashscope.legacy_1x.client.base_api'
""",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("old_name", "canonical_name"),
    [
        ("dashscope.agentstudio", "dashscope.legacy_1x.agentstudio"),
        (
            "dashscope.agentstudio.types.models",
            "dashscope.applications.agentstudio.types.models",
        ),
        (
            "dashscope.finetune.agentic_rl",
            "dashscope.legacy_1x.finetune.agentic_rl",
        ),
        (
            "dashscope.finetune.reinforcement.common.errors",
            "dashscope.legacy_1x.finetune.reinforcement.common.errors",
        ),
    ],
)
def test_historical_paths_select_the_1x_facades(old_name, canonical_name):
    assert importlib.import_module(old_name) is importlib.import_module(
        canonical_name,
    )


def test_agentic_rl_historical_contract_is_unchanged():
    module = importlib.import_module("dashscope.finetune.agentic_rl")
    for verb in ("cancel", "list", "get", "delete", "logs"):
        assert hasattr(module.AgenticRL, verb)
    assert module.AgenticRL.__module__.startswith("dashscope.legacy_1x")


def test_plain_import_preserves_cli_and_assistants_laziness():
    _run_python(
        "-c",
        """
import sys
import dashscope

names = (
    'dashscope.cli',
    'dashscope.legacy_1x.cli',
    'dashscope.assistants',
    'dashscope.legacy_1x.assistants',
)
assert not any(name in sys.modules for name in names)
""",
    )


def test_historical_cli_module_entrypoint():
    _run_python("-m", "dashscope.cli", "--help")
