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
        import cv2  # noqa: F401

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
            "错误: 未找到摄像头依赖。\n"
            "  一次性安装所有可选依赖: pip install 'acli[all]'\n"
            "  仅安装摄像头依赖:      pip install 'acli[camera]'\n"
            "  macOS 备用方案:        brew install imagesnap"
        )

    if backend == "opencv":
        return _capture_opencv(output_path)
    elif backend == "imagesnap":
        return _capture_imagesnap(output_path)
    return "错误: 未知的摄像头后端"


def _capture_opencv(output_path: str) -> str:
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "错误: 无法打开摄像头"

    # Warm up the camera (first few frames may be dark)
    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return "错误: 无法从摄像头读取画面"

    cv2.imwrite(output_path, frame)
    abs_path = os.path.abspath(output_path)
    size = os.path.getsize(abs_path)
    return f"已拍照保存: {abs_path} ({size / 1024:.1f} KB)"


def _capture_imagesnap(output_path: str) -> str:
    try:
        result = subprocess.run(
            ["imagesnap", "-w", "1.0", output_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"错误: imagesnap 失败 - {result.stderr.strip()}"
        abs_path = os.path.abspath(output_path)
        size = os.path.getsize(abs_path)
        return f"已拍照保存: {abs_path} ({size / 1024:.1f} KB)"
    except subprocess.TimeoutExpired:
        return "错误: 摄像头超时"
    except FileNotFoundError:
        return "错误: imagesnap 未安装 (brew install imagesnap)"


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------


def record(output_path: str = "camera_record.mp4", duration: float = 5.0) -> str:
    """Record video from webcam for *duration* seconds. Returns success/error message."""
    output_path = os.path.expanduser(output_path)
    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    ok, backend = is_available()
    if not ok:
        return (
            "错误: 未找到摄像头依赖。\n"
            "  一次性安装所有可选依赖: pip install 'acli[all]'\n"
            "  仅安装摄像头依赖:      pip install 'acli[camera]'\n"
            "  macOS 备用方案:        brew install imagesnap"
        )

    if backend == "opencv":
        return _record_opencv(output_path, duration)
    elif backend == "imagesnap":
        return _record_ffmpeg(output_path, duration)
    return "错误: 未知的摄像头后端"


def _record_opencv(output_path: str, duration: float) -> str:
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "错误: 无法打开摄像头"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return "错误: 无法创建视频文件"

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
        return "错误: 未录到任何画面"

    abs_path = os.path.abspath(output_path)
    size = os.path.getsize(abs_path)
    actual_dur = time.time() - start
    return (
        f"已录制保存: {abs_path} ({size / 1024:.1f} KB, {actual_dur:.1f}s, {frames} 帧)"
    )


def _record_ffmpeg(output_path: str, duration: float) -> str:
    """Fallback: use ffmpeg with avfoundation on macOS."""
    if not shutil.which("ffmpeg"):
        return "错误: 录制视频需要 opencv 或 ffmpeg (brew install ffmpeg)"
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
        )
        if result.returncode != 0:
            return f"错误: ffmpeg 失败 - {result.stderr.strip()[:200]}"
        abs_path = os.path.abspath(output_path)
        size = os.path.getsize(abs_path)
        return f"已录制保存: {abs_path} ({size / 1024:.1f} KB, {duration:.1f}s)"
    except subprocess.TimeoutExpired:
        return "错误: 录制超时"
