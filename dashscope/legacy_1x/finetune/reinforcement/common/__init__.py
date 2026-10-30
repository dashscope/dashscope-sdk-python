# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Historical common namespace; only errors are version-specific."""

import importlib

from dashscope.legacy_1x.finetune.reinforcement.common import errors

__all__ = ["errors"]


def __getattr__(name):
    if name == "errors":
        return errors
    try:
        module = importlib.import_module(
            f"dashscope.applications.finetune.reinforcement.common.{name}",
        )
    except ModuleNotFoundError as exc:
        raise AttributeError(name) from exc
    globals()[name] = module
    return module
