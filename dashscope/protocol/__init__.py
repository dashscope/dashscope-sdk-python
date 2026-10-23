# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Backward-compat shim; the implementation moved to ``dashscope.legacy_1x.protocol``.

Import from ``dashscope.legacy_1x.protocol`` (or the public ``dashscope``
root re-exports) in new code.
"""

from dashscope.legacy_1x._shim import install as _install

_install(__name__, "dashscope.legacy_1x.protocol")
