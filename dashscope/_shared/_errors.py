# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The one exception base class both stacks inherit from.

``except dashscope.legacy_1x.common.error.DashScopeException`` has to catch errors
raised by the 1.x API classes *and* by the AgentStudio sub-SDK, which the
2.0 client mounts directly. That only works if every subclass in both
stacks derives from the same class object, so the base lives here rather
than in either stack: copying it would silently split the hierarchy and
let errors escape a caller's ``except`` clause.

``dashscope.legacy_1x.common.error`` re-exports this name, so the 1.x import path
keeps working and keeps the identical class.
"""


class DashScopeException(Exception):
    """Base class for every error the dashscope SDK raises."""


__all__ = ["DashScopeException"]
