# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility re-export; the implementation moved to ``dashscope._shared``.

The session id state lives only in ``dashscope._shared._sdk_session`` so a
process keeps exactly one id. Import from there in new code.
"""

from dashscope._shared._sdk_session import sdk_session_id

__all__ = ["sdk_session_id"]
