# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The dashscope 1.x SDK.

Everything under this package is the legacy 1.x business surface. It is
kept importable both at its canonical ``dashscope.legacy_1x.*`` path and,
for backward compatibility, at the historical top-level paths (for
example ``dashscope.aigc`` -> ``dashscope.legacy_1x.aigc``) via thin
alias shims installed at the package root.

New code should target the 2.0 SDK, which lives at the top level of
``dashscope`` (``dashscope.Client``, ``dashscope.chat``, ...).
"""
