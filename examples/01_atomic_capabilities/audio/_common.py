# -*- coding: utf-8 -*-
"""Local PCM capture, WAV containers, and operating-system playback."""

import argparse
import math
import shutil
import subprocess
import sys
import wave
from contextlib import contextmanager
from pathlib import Path


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("秒数必须是大于 0 的有限数值")
    return seconds


@contextmanager
def open_wav(path, sample_rates=(8000, 16000)):
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
            or audio.getframerate() not in sample_rates
        ):
            raise ValueError(
                f"需要单声道、16-bit PCM WAV，采样率为 {sample_rates}；"
                f"实际格式：{audio.getparams()}。可先使用 ffmpeg 转换。"
            )
        if not audio.getnframes():
            raise ValueError("WAV 文件没有音频帧")
        yield audio


@contextmanager
def wav_output(path, sample_rate):
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError("PCM 输出必须使用 .wav 扩展名")
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            yield audio


def input_device(value):
    return int(value) if value.isdecimal() else value


def load_sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "录音需要 sounddevice 和 PortAudio；请在仓库根目录运行 "
            "python -m pip install -e '.[acli-voice]'。"
            "Linux 还需安装系统 PortAudio 库。"
        ) from error
    return sounddevice


@contextmanager
def microphone(sample_rate, device=None):
    sd = load_sounddevice()
    sd.check_input_settings(
        device=device, channels=1, dtype="int16", samplerate=sample_rate,
    )
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=sample_rate // 10,
        device=device,
        channels=1,
        dtype="int16",
    ) as stream:
        yield stream


def player_command(path):
    path = str(Path(path).resolve())
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay", path]
    if sys.platform == "win32" and shutil.which("powershell"):
        return [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "$p = New-Object System.Media.SoundPlayer; "
            "$p.SoundLocation = $env:DASHSCOPE_SAMPLE_WAV; $p.PlaySync()",
        ]
    if shutil.which("aplay"):
        return ["aplay", path]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path]
    raise RuntimeError(
        "找不到 WAV 播放器（afplay / PowerShell / aplay / ffplay）；"
        "可使用 --no-play 仅保存文件。"
    )


def play_wav(path):
    import os

    command = player_command(path)
    environment = dict(os.environ, DASHSCOPE_SAMPLE_WAV=str(Path(path).resolve()))
    process = subprocess.Popen(command, env=environment)
    try:
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
