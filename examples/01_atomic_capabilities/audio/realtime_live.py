# -*- coding: utf-8 -*-
"""Live 简洁示例：发送文字或音频文件，实时打印回答转写；Ctrl+C 退出。

设置 LIVE_API_KEY 或 DASHSCOPE_API_KEY；密钥须有目标环境访问权限。
默认复用已实测跑通的预发环境，可用 LIVE_WS_URL / LIVE_ROUTE_MODEL /
LIVE_FRONTEND_MODEL / LIVE_BACKEND_MODEL 覆写：路由模型决定接入地址，前端模型
处理语音，设置后端模型才委派 responses 后台推理。服务端必填 context.compaction，
缺失会拒绝初始化。WAV / PCM 文件需为 16 kHz、单声道 PCM16LE。
音频按 100 ms 帧发送，不追赶补发；本示例不采集或播放音频。
回答后连接保持打开；连续 30 秒未收到事件会报 SDK 超时异常。
运行：python -m examples.01_atomic_capabilities.audio.realtime_live [audio_file]
"""

import argparse
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from dashscope import Client

WS_URL = os.getenv(
    "LIVE_WS_URL", "wss://poc-01.cn-beijing.pre-maas.aliyuncs.com/api-ws/v1/realtime",
)
ROUTE_MODEL = os.getenv("LIVE_ROUTE_MODEL", "pre-qwen-live-test")
FRONTEND_MODEL = os.getenv("LIVE_FRONTEND_MODEL", "qwen-audio-realtime")
BACKEND_MODEL = os.getenv("LIVE_BACKEND_MODEL", "")


def read_audio(path):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1 or audio.getsampwidth() != 2
                or audio.getframerate() != 16000 or audio.getcomptype() != "NONE"
            ):
                raise ValueError("WAV 必须是 16 kHz、单声道、16-bit PCM")
            pcm = audio.readframes(audio.getnframes())
            if len(pcm) != audio.getnframes() * 2:
                raise ValueError("WAV 音频数据被截断")
    elif path.suffix.lower() == ".pcm":
        pcm = path.read_bytes()  # 裸 PCM 无格式信息，由调用者保证 16 kHz 单声道。
    else:
        raise ValueError("仅支持 .wav 或 .pcm 音频文件")
    if not pcm or len(pcm) % 2:
        raise ValueError("音频必须包含非空、偶数字节的 PCM16LE 数据")
    return pcm


def send_audio(session, pcm, stopped):
    pcm += bytes(32000)  # 一秒静音，让服务端 VAD 检测到本轮语音结束。
    for offset in range(0, len(pcm), 3200):
        if stopped.is_set():
            return
        session.append_audio(pcm[offset:offset + 3200].ljust(3200, b"\0"))
        if stopped.wait(0.1):  # 每帧 100 ms；延迟后不追赶补发。
            return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", nargs="?", help="16 kHz 单声道 PCM WAV / PCM 文件")
    args = parser.parse_args(argv)
    pcm = read_audio(args.audio_file) if args.audio_file else None
    delegation = {"type": "none"} if not BACKEND_MODEL else {
        "type": "responses",
        "responses": {
            "model": BACKEND_MODEL,
            "instructions": "直接回答用户问题，不使用外部工具",
        },
    }
    stopped = Event()
    try:
        with Client(
            api_key=os.getenv("LIVE_API_KEY"),
        ) as client, client.audio.live.connect(
            model=ROUTE_MODEL, ws_url=WS_URL, timeout=30,
        ) as session, ThreadPoolExecutor(max_workers=1) as pool:
            sender = None
            try:
                session.update_session(
                    model=FRONTEND_MODEL,
                    delegation=delegation,
                    context={"compaction": {"enabled": False, "mode": "local"}},
                    instructions="请用简短的中文回答。",
                    input_audio_format="pcm",
                    output_audio_format="pcm",
                )
                sent = False
                for event in session.events():
                    if event["type"] == "session.updated" and not sent:
                        sent = True
                        if pcm is None:
                            session.append_commentary("你好，请用一句话介绍自己。")
                            session.create_response()  # 纯文本不会自动触发作答。
                        else:
                            sender = pool.submit(send_audio, session, pcm, stopped)
                    elif event["type"] == "session.output_transcript.delta":
                        print(event["delta"], end="", flush=True)
                    elif event["type"] == "error":
                        error = event["error"]
                        message = (
                            error.get("message") or error.get("code") or "Live error"
                        )
                        print(message, file=sys.stderr)
            finally:
                stopped.set()
                session.close()  # 先中断 I/O，再等待发送线程退出。
            if sender is not None:
                sender.result()
    except KeyboardInterrupt:
        pass  # with 会释放连接；Ctrl+C 是本地主动退出，不是异常断连。
    print()


if __name__ == "__main__":
    main()
