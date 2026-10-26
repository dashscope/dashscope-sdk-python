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
    'rerank', 'threads', 'utils',
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


@pytest.mark.parametrize(
    ("old_name", "canonical_name"),
    [
        ("dashscope.agentstudio", "dashscope.applications.agentstudio"),
        (
            "dashscope.agentstudio.types.models",
            "dashscope.applications.agentstudio.types.models",
        ),
        (
            "dashscope.finetune.agentic_rl",
            "dashscope.applications.finetune.agentic_rl",
        ),
        (
            "dashscope.finetune.reinforcement.common.errors",
            "dashscope.applications.finetune.reinforcement.common.errors",
        ),
    ],
)
def test_application_public_paths_are_aliases(old_name, canonical_name):
    assert importlib.import_module(old_name) is importlib.import_module(
        canonical_name,
    )


def test_agentic_rl_current_contract_is_unchanged():
    module = importlib.import_module("dashscope.finetune.agentic_rl")
    assert not hasattr(module.AgenticRL, "list")
    assert module.SubmitResult is not None


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
