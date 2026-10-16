# -*- coding: utf-8 -*-
"""TingWu offline task call example."""

import os

from dashscope.multimodal.tingwu.tingwu import TingWu

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get(
    "DASHSCOPE_MODEL",
    "tingwu-automotive-service-inspection",
)


if __name__ == "__main__":
    # Create a TingWu task: pass user_defined_input for your
    # business scenario; fileUrl is the URL of the audio/video file
    # to analyze, and appid is the TingWu app ID
    response = TingWu.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        user_defined_input={
            "fileUrl": "http://demo.com/test.mp3",
            "appid": "123456",
        },
    )
    print(response)
