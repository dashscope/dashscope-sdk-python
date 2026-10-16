# -*- coding: utf-8 -*-
"""CV model call example: portrait style repaint
(wanx-style-repaint), including local image upload.

API doc: https://help.aliyun.com/zh/model-studio/portrait-style-redraw-api-reference
"""

import os

from dashscope.client.base_api import BaseAsyncApi
from dashscope.utils.oss_utils import check_and_upload_local

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "wanx-style-repaint-v1")

# Local image path; override via the DASHSCOPE_DEMO_IMAGE
# environment variable
IMAGE_PATH = os.environ.get(
    "DASHSCOPE_DEMO_IMAGE",
    os.path.expanduser("~/Downloads/cat.png"),
)


if __name__ == "__main__":
    api_key = os.getenv("DASHSCOPE_API_KEY")

    # Upload the local file to temporary OSS storage first to get a URL
    uploaded, image_url = check_and_upload_local(
        model=MODEL,
        content=IMAGE_PATH,
        api_key=api_key,
    )

    kwargs = {}
    if uploaded is True:
        kwargs["headers"] = {"X-DashScope-OssResourceResolve": "enable"}

    response = BaseAsyncApi.call(
        model=MODEL,
        input={
            "image_url": image_url,
            "style_index": 3,
        },
        task_group="aigc",
        task="image-generation",
        function="generation",
        **kwargs,
    )
    print(f"response:\n{response}")
