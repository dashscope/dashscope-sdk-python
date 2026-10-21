# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Request-parameter rules that depend on the model name."""

from __future__ import annotations

# tts / omni / qwen-deep-research honor incremental_output=False on the
# server; every other model streams incrementally only, so the SDK has to
# keep the wire incremental and assemble the cumulative view itself.
_SERVER_SIDE_MODELS = ("tts", "omni", "qwen-deep-research")


def should_modify_incremental_output(model_name: str) -> bool:
    """Whether the SDK must build cumulative frames client-side.

    ``False`` for the models that honor ``incremental_output`` server-side.
    """
    if not isinstance(model_name, str):
        return True
    lowered = model_name.lower()
    return not any(name in lowered for name in _SERVER_SIDE_MODELS)
