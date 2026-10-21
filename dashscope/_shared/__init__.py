# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Neutral leaf layer shared by the 1.x API and the 2.0 implementation.

Modules here hold state or values that must exist exactly once per
installed distribution (the process-wide session id, the version
number). Copying them into either stack would silently split that single
source of truth, so both stacks import from here instead.

Nothing in this package may import ``dashscope`` or any of its
subpackages: it sits below both stacks and is the anchor that keeps them
from forming an import cycle. Stdlib only.
"""
