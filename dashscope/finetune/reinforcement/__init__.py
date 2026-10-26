# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Legacy alias for :mod:`dashscope.reinforcement`.

The public compatibility package owns the lazy module aliasing; this path
only registers its prefix so existing dotted-path consumers keep resolving
to the same v2 module objects.
"""

# Import the v2 exports directly rather than wildcard-importing the public
# shim, whose __all__ includes the deliberately lazy CLI ``app``.
from dashscope import reinforcement as _public
from dashscope.v2.finetune.reinforcement import *  # noqa: F401,F403
from dashscope.v2.finetune.reinforcement import (  # noqa: F401
    common,
    component,
)

__all__ = list(_public.__all__)


def __getattr__(name):
    if name == "app":
        return _public.app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_public._register_alias(__name__)
