# -*- coding: utf-8 -*-
from http import HTTPStatus
from dashscope import ImageSynthesis
import os

prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
api_key = os.getenv("DASHSCOPE_API_KEY")


def simple_call():
    print("----sync call, please wait a moment----")
    rsp = ImageSynthesis.call(
        api_key=api_key,
        model="wanx2.1-t2i-turbo",
        prompt=prompt,
        n=1,
        size="1024*1024",
    )
    if rsp.status_code == HTTPStatus.OK:
        print("response: %s" % rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


def sync_call():
    print("----sync call, please wait a moment----")
    """
    Note: This method currently now only supports wan2.2-t2i-flash and wan2.2-t2i-plus.
        Using other models will result in an error，More raw image models may be added for use later
    """
    rsp = ImageSynthesis.sync_call(
        api_key=api_key,
        model="wan2.2-t2i-flash",
        prompt=prompt,
        n=1,
        size="1024*1024",
    )
    if rsp.status_code == HTTPStatus.OK:
        print("response: %s" % rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


if __name__ == "__main__":
    # simple_call()
    sync_call()
