# -*- coding: utf-8 -*-
"""Camera tools — allows the LLM to capture photos and record video
from the webcam."""

from __future__ import annotations

from dashscope.acli.tools.registry import PermissionLevel, tool


@tool(
    name="camera_capture",
    description="用摄像头拍一张照片并保存到指定路径。拍完后用户可以用 @路径 引用图片发给视觉模型分析。",
    permission=PermissionLevel.CONFIRM,
)
def camera_capture(path: str = "camera_capture.jpg") -> str:
    from dashscope.acli.ui.camera import capture

    return capture(path)


@tool(
    name="camera_record",
    description="用摄像头录制一段视频并保存到指定路径。可指定录制时长（秒）。",
    permission=PermissionLevel.CONFIRM,
)
def camera_record(
    path: str = "camera_record.mp4",
    duration: float = 5.0,
) -> str:
    from dashscope.acli.ui.camera import record

    return record(path, duration)
