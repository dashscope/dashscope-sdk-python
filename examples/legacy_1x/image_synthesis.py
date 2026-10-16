# -*- coding: utf-8 -*-
"""ImageSynthesis text-to-image example (wanx series)."""

import os
from http import HTTPStatus

from dashscope import ImageSynthesis

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "wanx2.1-t2i-turbo")

PROMPT = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"


def print_response(response) -> None:
    if response.status_code == HTTPStatus.OK:
        for result in response.output.results:
            print(f"image url: {result['url']}")
    else:
        print(
            f"Failed, status_code: {response.status_code}, "
            f"code: {response.code}, message: {response.message}",
        )


if __name__ == "__main__":
    # Async-task mode: the SDK polls until the task completes
    print("----- async task request (call) -----")
    rsp = ImageSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        prompt=PROMPT,
        n=1,
        size="1024*1024",
    )
    print_response(rsp)

    # Sync-API mode: note that only wan2.2-t2i-flash and
    # wan2.2-t2i-plus are currently supported
    print("----- sync request (sync_call) -----")
    rsp = ImageSynthesis.sync_call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="wan2.2-t2i-flash",
        prompt=PROMPT,
        n=1,
        size="1024*1024",
    )
    print_response(rsp)
