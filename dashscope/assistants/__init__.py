# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import warnings

# yapf: disable

from dashscope.assistants.assistant_types import (
    Assistant, AssistantFile,
    AssistantList,
    DeleteResponse,
)
from dashscope.assistants.assistants import Assistants

__all__ = [
    'Assistant',
    'Assistants',
    'AssistantList',
    'AssistantFile',
    'DeleteResponse',
]

# Deprecation warning
_DEPRECATION_MSG = (
    "The Assistants API (dashscope.assistants) is deprecated and will be "
    "removed in a future release. Please migrate to the Responses API. "
    "See https://help.aliyun.com/zh/model-studio/"
    "synchronous-call-api-reference for migration details."
)

warnings.warn(_DEPRECATION_MSG, category=DeprecationWarning, stacklevel=2)
