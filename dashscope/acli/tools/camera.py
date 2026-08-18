# -*- coding: utf-8 -*-
"""Camera tools — allows the LLM to capture photos and record video
from the webcam."""

from __future__ import annotations

from dashscope.acli.tools.registry import PermissionLevel, tool


@tool(
    name="camera_capture",
    description=(
        "Take a photo with the webcam and save it to the given path. "
        "The user can then reference it via @path to send it to a "
        "vision model for analysis."
    ),
    permission=PermissionLevel.CONFIRM,
)
def camera_capture(path: str = "camera_capture.jpg") -> str:
    from dashscope.acli.ui.camera import capture

    return capture(path)


@tool(
    name="camera_record",
    description=(
        "Record a video clip with the webcam and save it to the given "
        "path. The recording duration (seconds) can be specified."
    ),
    permission=PermissionLevel.CONFIRM,
)
def camera_record(
    path: str = "camera_record.mp4",
    duration: float = 5.0,
) -> str:
    from dashscope.acli.ui.camera import record

    return record(path, duration)
