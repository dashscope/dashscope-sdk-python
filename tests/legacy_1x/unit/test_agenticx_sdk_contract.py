# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import importlib
import inspect
import sys
from pathlib import Path

import pytest

import dashscope

_REPO = Path(__file__).resolve().parents[3]
_ACLI_EXTRAS = frozenset(
    {
        "acli",
        "acli-all",
        "acli-anthropic",
        "acli-camera",
        "acli-openai",
        "acli-voice",
    },
)


def test_vendored_acli_is_removed():
    assert not (_REPO / "dashscope" / "acli").exists()


def test_tts_v2_exposes_the_names_agenticx_uses():
    module = importlib.import_module("dashscope.audio.tts_v2")
    for name in ("AudioFormat", "ResultCallback", "SpeechSynthesizer"):
        assert hasattr(module, name)


def test_asr_exposes_the_names_agenticx_uses():
    module = importlib.import_module("dashscope.audio.asr")
    for name in ("Recognition", "RecognitionCallback"):
        assert hasattr(module, name)


def test_root_api_key_is_readable_and_writable(monkeypatch):
    assert hasattr(dashscope, "api_key")
    monkeypatch.setattr(dashscope, "api_key", "sk-agenticx-contract")
    assert dashscope.api_key == "sk-agenticx-contract"


def test_setup_declares_the_acli_extras_and_agenticx_dependency():
    source = (_REPO / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source, "setup.py")
    groups = None
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "extras_require":
            groups = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
            }
    assert groups is not None
    assert not _ACLI_EXTRAS - groups
    assert source.count("dashscope-agenticx[tui]>=1.0.0,<2") == 2


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="dashscope-agenticx requires Python 3.10+",
)
def test_agenticx_embedded_api_matches_dashscope_adapter():
    embedded = importlib.import_module("dashscope_agenticx.ui.embedded")
    parameters = inspect.signature(embedded.run).parameters
    assert {
        "app_name",
        "prompt_symbol",
        "api_key",
        "command",
        "tui",
        "module",
        "guide_url",
    } <= parameters.keys()


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="dashscope-agenticx requires Python 3.10+",
)
def test_agenticx_expert_downloader_matches_dashscope_adapter():
    examples = importlib.import_module("dashscope_agenticx.cli.examples")
    assert callable(examples._handle_example_command)
    assert (
        "args"
        in inspect.signature(
            examples._handle_example_command,
        ).parameters
    )
    assert examples._DEFAULT_EXAMPLES_REPO == (
        "https://github.com/dashscope/dashscope-agenticx-experts.git"
    )
