# -*- coding: utf-8 -*-
"""VideoSynthesis video-generation example (wan series).

Runs the reference-video-to-video (r2v) scenario by default;
uncomment the other scenarios to run them.
Note: video-generation tasks take a while; please be patient.
"""

import os
from http import HTTPStatus

from dashscope import VideoSynthesis

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
PROMPT = "一只小猫在月光下奔跑"

REFERENCE_URLS = [
    "https://test-data-center.oss-accelerate.aliyuncs.com/"
    "wanx/video/resources/with_human_voice_11s.mov",
]

MEDIA_I2V = [
    {
        "type": VideoSynthesis.MediaType.LAST_FRAME,
        "url": "https://wanx.alicdn.com/material/20250318/last_frame.png",
    },
    {
        "type": VideoSynthesis.MediaType.FIRST_FRAME,
        "url": "https://wanx.alicdn.com/material/20250318/first_frame.png",
    },
    {
        "type": VideoSynthesis.MediaType.DRIVING_AUDIO,
        "url": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/audio/mp3_3s.mp3",
    },
]

MEDIA_R2V = [
    {
        "type": VideoSynthesis.MediaType.REFERENCE_IMAGE,
        "url": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/image/res240_269.jpg",
    },
    {
        "type": VideoSynthesis.MediaType.REFERENCE_IMAGE,
        "url": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/image/man_5K_7_7K_18_4M.JPG",
        "reference_voice": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/audio/2s.wav",
    },
    {
        "type": VideoSynthesis.MediaType.REFERENCE_VIDEO,
        "url": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/video/resources/cast/100M.mov",
        "reference_voice": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/audio/mp3_1s.mp3",
    },
    {
        "type": VideoSynthesis.MediaType.REFERENCE_VIDEO,
        "url": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/video/resources/cast/29_99s.mp4",
        "reference_description": "这是一个身穿蓝衣的男子<cast>,他有着浓密的络腮胡",
    },
    {
        "type": VideoSynthesis.MediaType.REFERENCE_VIDEO,
        "url": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/video/resources/cast/cat_127.mp4",
        "reference_voice": "https://test-data-center.oss-accelerate.aliyuncs.com/"
        "wanx/audio/wav_10s.wav",
        "reference_description": "这是一只毛绒小猫<cast>,它正在对着镜头微笑",
    },
]


def print_response(rsp) -> None:
    if rsp.status_code == HTTPStatus.OK:
        print(f"video url: {rsp.output.video_url}")
    else:
        print(
            f"Failed, status_code: {rsp.status_code}, "
            f"code: {rsp.code}, message: {rsp.message}",
        )


def call_r2v() -> None:
    """Reference video to video (wan2.6-r2v)."""
    print("----- r2v request (wan2.6-r2v) -----")
    rsp = VideoSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="wan2.6-r2v",
        reference_urls=REFERENCE_URLS,
        shot_type="multi",
        audio=True,
        watermark=True,
        prompt=PROMPT,
    )
    print_response(rsp)


def call_t2v() -> None:
    """Text to video (wan2.7-t2v)."""
    print("----- t2v request (wan2.7-t2v) -----")
    rsp = VideoSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="wan2.7-t2v",
        audio=True,
        watermark=True,
        prompt=PROMPT,
    )
    print_response(rsp)


def call_i2v() -> None:
    """First/last-frame to video (wan2.7-i2v), with an optional
    driving audio."""
    print("----- i2v request (wan2.7-i2v) -----")
    rsp = VideoSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="wan2.7-i2v",
        media=MEDIA_I2V,
        audio=True,
        watermark=True,
        prompt=PROMPT,
    )
    print_response(rsp)


def call_r2v_wan27() -> None:
    """Reference-material to video (wan2.7-r2v), supporting multiple
    reference images/videos and voice-overs."""
    print("----- r2v request (wan2.7-r2v) -----")
    rsp = VideoSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="wan2.7-r2v",
        media=MEDIA_R2V,
        audio=True,
        watermark=True,
        prompt=PROMPT,
    )
    print_response(rsp)


if __name__ == "__main__":
    call_r2v()
    # call_t2v()
    # call_i2v()
    # call_r2v_wan27()
