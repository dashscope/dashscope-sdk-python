# -*- coding: utf-8 -*-
import os

from dashscope.client.base_api import BaseAsyncApi
from dashscope.utils.oss_utils import check_and_upload_local

# style repaint (ref: https://help.aliyun.com/zh/model-studio/portrait-style-redraw-api-reference)

api_key = os.getenv("DASHSCOPE_API_KEY")
model = "wanx-style-repaint-v1"
file_path = "~/Downloads/cat.png"

uploaded, image_url = check_and_upload_local(
    model=model,
    content=file_path,
    api_key=api_key,
)

kwargs = {}
if uploaded is True:
    headers = {"X-DashScope-OssResourceResolve": "enable"}
    kwargs["headers"] = headers

response = BaseAsyncApi.call(
    model=model,
    input={
        # "image_url": "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/public/dashscope/test.png",
        "image_url": image_url,
        "style_index": 3,
    },
    task_group="aigc",
    task="image-generation",
    function="generation",
    **kwargs,
)

print("response: \n%s\n" % response)
