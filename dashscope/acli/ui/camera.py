# -*- coding: utf-8 -*-
"""Camera capture: take a photo from the webcam."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time


def is_available() -> tuple[bool, str]:
    """Check if camera capture is possible."""
    try:
        import cv2  # noqa: F401  # pylint: disable=unused-import

        return True, "opencv"
    except ImportError:
        pass
    if platform.system() == "Darwin" and shutil.which("imagesnap"):
        return True, "imagesnap"
    return False, ""


def capture(output_path: str = "camera_capture.jpg") -> str:
    """Capture a single frame from webcam. Returns success/error message."""
    output_path = os.path.expanduser(output_path)
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    ok, backend = is_available()
    if not ok:
        return (
            "Error: camera dependencies not found.\n"
            "  Install all optional deps: pip install 'acli[all]'\n"
            "  Install camera deps only:  pip install 'acli[camera]'\n"
            "  macOS fallback:            brew install imagesnap"
        )

    if backend == "opencv":
        return _capture_opencv(output_path)
    elif backend == "imagesnap":
        return _capture_imagesnap(output_path)
    return "Error: unknown camera backend"


def _capture_opencv(output_path: str) -> str:
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: cannot open camera"

    # Warm up the camera (first few frames may be dark)
    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return "Error: cannot read frame from camera"

    cv2.imwrite(output_path, frame)
    abs_path = os.path.abspath(output_path)
    size = os.path.getsize(abs_path)
    return f"Photo saved: {abs_path} ({size / 1024:.1f} KB)"


def _capture_imagesnap(output_path: str) -> str:
    try:
        result = subprocess.run(
            ["imagesnap", "-w", "1.0", output_path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return f"Error: imagesnap failed - {result.stderr.strip()}"
        abs_path = os.path.abspath(output_path)
        size = os.path.getsize(abs_path)
        return f"Photo saved: {abs_path} ({size / 1024:.1f} KB)"
    except subprocess.TimeoutExpired:
        return "Error: camera timed out"
    except FileNotFoundError:
        return "Error: imagesnap not installed (brew install imagesnap)"


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------


def record(
    output_path: str = "camera_record.mp4",
    duration: float = 5.0,
) -> str:
    """Record video from webcam for *duration* seconds. Returns
    success/error message."""
    output_path = os.path.expanduser(output_path)
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    ok, backend = is_available()
    if not ok:
        return (
            "Error: camera dependencies not found.\n"
            "  Install all optional deps: pip install 'acli[all]'\n"
            "  Install camera deps only:  pip install 'acli[camera]'\n"
            "  macOS fallback:            brew install imagesnap"
        )

    if backend == "opencv":
        return _record_opencv(output_path, duration)
    elif backend == "imagesnap":
        return _record_ffmpeg(output_path, duration)
    return "Error: unknown camera backend"


def _record_opencv(output_path: str, duration: float) -> str:
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: cannot open camera"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return "Error: cannot create video file"

    start = time.time()
    frames = 0
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frames += 1

    writer.release()
    cap.release()

    if frames == 0:
        return "Error: no frames recorded"

    abs_path = os.path.abspath(output_path)
    size = os.path.getsize(abs_path)
    actual_dur = time.time() - start
    return (
        f"Recording saved: {abs_path} ({size / 1024:.1f} KB, "
        f"{actual_dur:.1f}s, {frames} frames)"
    )


def _record_ffmpeg(output_path: str, duration: float) -> str:
    """Fallback: use ffmpeg with avfoundation on macOS."""
    if not shutil.which("ffmpeg"):
        return (
            "Error: video recording needs opencv or ffmpeg "
            "(brew install ffmpeg)"
        )
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "avfoundation",
                "-framerate",
                "30",
                "-i",
                "0",
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=duration + 10,
            check=False,
        )
        if result.returncode != 0:
            return f"Error: ffmpeg failed - {result.stderr.strip()[:200]}"
        abs_path = os.path.abspath(output_path)
        size = os.path.getsize(abs_path)
        return (
            f"Recording saved: {abs_path} ({size / 1024:.1f} KB, "
            f"{duration:.1f}s)"
        )
    except subprocess.TimeoutExpired:
        return "Error: recording timed out"
