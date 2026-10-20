# -*- coding: utf-8 -*-
"""Qwen-Audio 单轮实时语音对话：发送 WAV 或麦克风音频，保存并播放回答。"""

import argparse
import base64
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dashscope import Client
from dashscope.v2.errors import DashScopeError
from ._common import (
    input_device,
    load_sounddevice,
    microphone,
    open_wav,
    play_wav,
    player_command,
    positive_seconds,
    wav_output,
)


def receive_until(session, target, deadline, output=None):
    size = 0
    while time.monotonic() < deadline:
        event = session.recv_event()
        if event is None:
            raise RuntimeError(f"连接已关闭，未收到 {target}")
        kind = event.get("type")
        if kind == "response.audio.delta" and output is not None:
            audio = base64.b64decode(event["delta"], validate=True)
            output.writeframesraw(audio)
            size += len(audio)
        elif kind in ("response.text.delta", "response.audio_transcript.delta"):
            print(event.get("delta", ""), end="", flush=True)
        elif kind in ("response.text.done", "response.audio_transcript.done"):
            print("\n完整回答：", event.get("text", event.get("transcript", "")))
        elif kind == "conversation.item.input_audio_transcription.completed":
            print("\n输入转写：", event.get("transcript", ""))
        elif kind == "response.done":
            response = event.get("response") or {}
            if response.get("status") != "completed":
                raise RuntimeError(f"回答未正常完成：{response.get('status')}")
            print("\nresponse_id:", response.get("id"))
            print("usage:", response.get("usage"))
        if kind == target:
            return size
    raise TimeoutError(f"等待 {target} 超过总时限")


def send_input(session, receiver, args, source, turn_deadline):
    sent = 0
    if source is not None:
        deadline = time.monotonic()
        while True:
            if time.monotonic() >= turn_deadline:
                raise TimeoutError("发送音频超过本轮总时限")
            audio = source.readframes(1600)
            if not audio:
                break
            if receiver.done():
                receiver.result()
                raise RuntimeError("输入尚未结束，接收线程已退出")
            session.append_audio(audio)
            sent += len(audio)
            deadline += len(audio) / 32000
            time.sleep(max(0, deadline - time.monotonic()))
        if sent != source.getnframes() * 2:
            raise ValueError("WAV 数据被截断，实际音频长度小于文件头声明")
    else:
        stream = microphone(16000, args.device)
        try:
            stream.start()
            print(f"正在录音，{args.seconds:g} 秒后结束；Ctrl+C 提前结束本轮。")
            remaining = round(args.seconds * 16000)
            try:
                while remaining > 0:
                    if time.monotonic() >= turn_deadline:
                        raise TimeoutError("录音超过本轮总时限")
                    if receiver.done():
                        receiver.result()
                        raise RuntimeError("录音尚未结束，接收线程已退出")
                    count = min(1600, remaining)
                    audio, overflow = stream.read(count)
                    if overflow:
                        raise RuntimeError("麦克风输入溢出，音频已丢失")
                    session.append_audio(bytes(audio))
                    sent += len(audio)
                    remaining -= count
            except KeyboardInterrupt:
                print("\n录音已结束，正在等待回答。")
        finally:
            try:
                stream.stop()
            finally:
                stream.close()
    if sent < 3200:
        raise ValueError("本轮至少需要 0.1 秒音频")


def make_parser():
    parser = argparse.ArgumentParser(
        description="Qwen-Audio 单轮实时语音对话：发送 WAV 或麦克风音频，保存并播放回答。",
        epilog=(
            "python -m examples.01_atomic_capabilities.audio.direct_call.realtime_qwen_audio --wav input.wav "
            "--output reply.wav\n"
            "python -m examples.01_atomic_capabilities.audio.direct_call.realtime_qwen_audio --microphone "
            "--seconds 10 --output reply.wav\n"
            "安装源码：python -m pip install -e '.[acli-voice]'。"
            "配置 DASHSCOPE_API_KEY / DASHSCOPE_WORKSPACE / DASHSCOPE_REGION。\n"
            "固定使用 qwen-audio-3.0-realtime-plus，手动结束单轮输入。"
            "本示例未进行云端验证；实际调用可能计费。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path, help="0.1～20 秒、16 kHz/16-bit/单声道 WAV")
    source.add_argument("--microphone", action="store_true", help="实际开启系统麦克风")
    parser.add_argument("--seconds", type=positive_seconds, default=10.0,
                        help="录音时长，0.1～20 秒，默认 10")
    parser.add_argument("--device", type=input_device, help="录音设备 ID 或名称")
    parser.add_argument("--voice", default="longanqian", help="音色，默认 longanqian")
    parser.add_argument("--instructions", default="请用简洁自然的中文回答。",
                        help="Qwen-Audio 回答指令")
    parser.add_argument("--output", type=Path, required=True, help="新的 24 kHz WAV 文件")
    parser.add_argument("--no-play", action="store_true", help="只保存回答，不播放")
    parser.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="单次 socket 操作超时（秒）")
    parser.add_argument("--turn-timeout", type=positive_seconds, default=120.0,
                        help="整轮等待预算；阻塞操作最多额外耗时 --timeout 秒")
    parser.add_argument("--region", help="服务地域")
    parser.add_argument("--workspace", help="工作空间 ID")
    return parser


def run_turn(args, source=None):
    raw, output = wav_output(args.output, 24000)
    try:
        client = Client(
            region=args.region, workspace=args.workspace, timeout=args.timeout,
        )
        try:
            session = client.audio.realtime.connect(
                model="qwen-audio-3.0-realtime-plus",
            )
            executor = None
            try:
                session.connect()
                deadline = time.monotonic() + args.turn_timeout
                session.update_session(
                    ["text", "audio"], voice=args.voice,
                    input_audio_format="pcm", output_audio_format="pcm",
                    enable_turn_detection=False, instructions=args.instructions,
                )
                receive_until(session, "session.updated", deadline)
                executor = ThreadPoolExecutor(max_workers=1)
                receiver = executor.submit(
                    receive_until, session, "response.done", deadline, output,
                )
                send_input(session, receiver, args, source, deadline)
                session.commit()
                session.create_response()
                size = receiver.result(timeout=max(0, deadline - time.monotonic()))
            finally:
                try:
                    session.close()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
            if not size or size % 2:
                raise RuntimeError(f"回答音频为空或不完整（{size} 字节）")
        finally:
            client.close()
    finally:
        try:
            output.close()
        finally:
            raw.close()
    return size


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.microphone and not 0.1 <= args.seconds <= 20:
        parser.error("--seconds 应在 0.1～20 之间")
    if not args.no_play:
        player_command(args.output)
    if args.wav:
        source = open_wav(args.wav, (16000,))
        try:
            duration = source.getnframes() / 16000
            if not 0.1 <= duration <= 20:
                parser.error("本示例的单轮 WAV 输入限制为 0.1～20 秒")
            size = run_turn(args, source)
        finally:
            source.close()
    else:
        load_sounddevice().check_input_settings(
            device=args.device, channels=1, dtype="int16", samplerate=16000,
        )
        size = run_turn(args)
    print(f"已保存 {args.output.resolve()}（{size / 48000:.2f} 秒）")
    if not args.no_play:
        play_wav(args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消；回答文件可能不完整。", file=sys.stderr)
        sys.exit(130)
    except subprocess.SubprocessError as error:
        print(f"回答音频已保存，但播放器失败：{error}", file=sys.stderr)
        sys.exit(1)
    except (DashScopeError, OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"对话失败：{error}；已创建的回答文件可能不完整。", file=sys.stderr)
        sys.exit(1)
