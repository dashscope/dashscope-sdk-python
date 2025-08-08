# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import platform
from abc import ABC, abstractmethod

from dashscope.common.constants import DASHSCOPE_DISABLE_DATA_INSPECTION_ENV
from dashscope.version import __version__


class BaseRequest(ABC):
    def __init__(self) -> None:
        try:
            platform_info = platform.platform()
        except Exception:
            platform_info = "unknown"

        try:
            processor_info = platform.processor()
        except Exception:
            processor_info = "unknown"

        ua = 'dashscope/%s; python/%s; platform/%s; processor/%s' % (
            __version__,
            platform.python_version(),
            platform_info,
            processor_info,
        )
        self.headers = {'user-agent': ua}
        disable_data_inspection = os.environ.get(
            DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, 'true')

        if (disable_data_inspection.lower() == 'false'):
            self.headers['X-DashScope-DataInspection'] = 'enable'

    @abstractmethod
    def call(self):
        raise NotImplementedError()


class AioBaseRequest(BaseRequest):
    @abstractmethod
    async def aio_call(self):
        raise NotImplementedError()
