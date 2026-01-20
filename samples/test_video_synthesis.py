# -*- coding: utf-8 -*-
from http import HTTPStatus
from dashscope import VideoSynthesis
import os

prompt = "一只小猫在月光下奔跑"
audio_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/ozwpvi/rap.mp3"
reference_video_urls = [
    "https://test-data-center.oss-accelerate.aliyuncs.com/wanx/video/resources/with_human_voice_11s.mov",
]
api_key = os.getenv("DASHSCOPE_API_KEY")


def simple_call():
    print("----sync call, please wait a moment----")
    rsp = VideoSynthesis.call(
        api_key=api_key,
        model="wan2.6-r2v",
        reference_video_urls=reference_video_urls,
        shot_type="multi",
        audio=True,
        watermark=True,
        prompt=prompt,
    )
    if rsp.status_code == HTTPStatus.OK:
        print("response: %s" % rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


if __name__ == "__main__":
    simple_call()
