# -*- coding: utf-8 -*-
"""Application (Model Studio app) call example: streaming output
with file input."""

import os
from http import HTTPStatus

from dashscope import Application

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), and the
# app ID via the DASHSCOPE_APP_ID environment variable
APP_ID = os.environ.get("DASHSCOPE_APP_ID", "your-app-id")


if __name__ == "__main__":
    print("----- streaming request -----")
    responses = Application.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        app_id=APP_ID,
        prompt="总结文件内容",
        stream=True,  # streaming output
        # has_thoughts=True,  # include node-level content
        incremental_output=True,
        file_list=[
            "https://dashscope.oss-cn-beijing.aliyuncs.com/"
            "audios/welcome.mp3",
        ],
        # flow_stream_mode='agent_format'  # Agent mode: expose node output
    )
    for response in responses:
        if response.status_code != HTTPStatus.OK:
            print(f"request_id={response.request_id}")
            print(f"code={response.status_code}")
            print(f"message={response.message}")
            print(
                "请参考文档：https://help.aliyun.com/zh/model-studio/"
                "developer-reference/error-code",
            )
        else:
            print(response.output.text, end="", flush=True)
    print()
