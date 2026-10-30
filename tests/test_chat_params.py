# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the model-name rule behind incremental_output.

The streaming behaviour this feeds is covered end to end in test_chat.py
(``test_incremental_output_false_merges_client_side`` and the exempt-model
cases); this pins the rule itself.
"""

from __future__ import annotations

import pytest

from dashscope.resources.chat._params import (
    should_modify_incremental_output,
)


@pytest.mark.parametrize(
    "model",
    [
        "qwen-tts",
        "CosyVoice-TTS-v1",
        "qwen-omni-turbo",
        "Qwen2.5-Omni-7B",
        "qwen-deep-research",
        "qwen-DEEP-Research",
    ],
)
def test_server_side_models_are_exempt(model):
    assert should_modify_incremental_output(model) is False


@pytest.mark.parametrize(
    "model",
    ["qwen-plus", "qwen-max", "qwen-vl-max", "qwen3-coder-plus", ""],
)
def test_incremental_only_models_merge_client_side(model):
    assert should_modify_incremental_output(model) is True


@pytest.mark.parametrize("model", [None, 123, ["qwen-tts"]])
def test_a_missing_model_name_merges_client_side(model):
    # Anything that is not a model name falls back to the common case
    # rather than guessing an exemption.
    assert should_modify_incremental_output(model) is True
