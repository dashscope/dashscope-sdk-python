# -*- coding: utf-8 -*-
"""Translate speech with Qwen3.8-LiveTranslate, saving the spoken translation."""

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
    responses = 0
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
            print("\n译文：", event.get("text", event.get("transcript", "")))
        elif kind == "conversation.item.input_audio_transcription.completed":
            print("\n原文：", event.get("transcript", ""))
        elif kind == "response.done":
            response = event.get("response") or {}
            if response.get("status") not in (None, "completed"):
                raise RuntimeError(f"翻译未正常完成：{response.get('status')}")
            responses += 1
            print("\nresponse_id:", response.get("id"))
            print("usage:", response.get("usage"))
        if kind == target:
            if target == "session.finished" and not responses:
                raise RuntimeError("会话结束，但未收到翻译响应结束事件")
            return size
    raise TimeoutError(f"等待 {target} 超过总时限")


def send_input(session, receiver, args, source, turn_deadline):
    sent = 0
    if source is not None:
        deadline = time.monotonic()
        while True:
            if time.monotonic() >= turn_deadline:
                raise TimeoutError("发送音频超过本次总时限")
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
            print(f"正在录音翻译，{args.seconds:g} 秒后结束；Ctrl+C 停止输入。")
            remaining = round(args.seconds * 16000)
            try:
                while remaining > 0:
                    if time.monotonic() >= turn_deadline:
                        raise TimeoutError("录音超过本次总时限")
                    if receiver.done():
                        receiver.result()
                        raise RuntimeError("录音尚未结束，接收线程已退出")
                    count = min(1600, remaining)
                    audio, overflow = stream.read(count)
                    if overflow:
                        raise RuntimeError("麦克风输入溢出，音频已丢失")
                    pcm = bytes(audio)
                    if len(pcm) != count * 2:
                        raise RuntimeError("麦克风返回的 PCM 帧不完整")
                    session.append_audio(pcm)
                    sent += len(pcm)
                    remaining -= count
            except KeyboardInterrupt:
                print("\n录音已结束，正在等待剩余译文。")
        finally:
            try:
                stream.stop()
            finally:
                stream.close()
    if sent < 3200:
        raise ValueError("本次至少需要 0.1 秒音频")


def make_parser():
    parser = argparse.ArgumentParser(
        description="Qwen3.8-LiveTranslate：边输入边翻译，保存译文音频后播放。",
        epilog=(
            "python -m examples.01_atomic_capabilities.audio.direct_call.realtime_translation --wav input.wav "
            "--language en --output translation.wav\n"
            "python -m examples.01_atomic_capabilities.audio.direct_call.realtime_translation --microphone "
            "--seconds 5 --language en --output translation.wav\n"
            "配置 DASHSCOPE_API_KEY / DASHSCOPE_WORKSPACE / DASHSCOPE_REGION。"
            "使用服务端默认说话人检测；示例未进行云端验证，实际调用可能计费。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path, help="0.1～20 秒、16 kHz/16-bit/单声道 WAV")
    source.add_argument("--microphone", action="store_true", help="实际开启系统麦克风")
    parser.add_argument("--seconds", type=positive_seconds, default=10.0,
                        help="录音时长，0.1～20 秒，默认 10")
    parser.add_argument("--device", type=input_device, help="录音设备 ID 或名称")
    parser.add_argument("--language", default="en", help="目标语种代码，默认 en（英语）")
    parser.add_argument("--output", type=Path, required=True, help="新的 24 kHz WAV 文件")
    parser.add_argument("--no-play", action="store_true", help="只保存，不播放")
    parser.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="单次 socket 操作超时（秒）")
    parser.add_argument("--turn-timeout", type=positive_seconds, default=120.0,
                        help="本次翻译总预算；阻塞操作最多额外耗时 --timeout 秒")
    parser.add_argument("--region")
    parser.add_argument("--workspace")
    return parser


def run_turn(args, source=None):
    raw, output = wav_output(args.output, 24000)
    try:
        client = Client(
            region=args.region, workspace=args.workspace, timeout=args.timeout,
        )
        try:
            session = client.audio.realtime.connect(
                model="qwen3.8-livetranslate-flash-realtime",
            )
            executor = None
            try:
                session.connect()
                deadline = time.monotonic() + args.turn_timeout
                session.update_session(
                    output_modalities=["text", "audio"],
                    translation_params={"language": args.language},
                )
                receive_until(session, "session.updated", deadline)
                executor = ThreadPoolExecutor(max_workers=1)
                receiver = executor.submit(
                    receive_until, session, "session.finished", deadline, output,
                )
                send_input(session, receiver, args, source, deadline)
                # Drain all translations, not just the first response.done.
                session.end_session()
                size = receiver.result(timeout=max(0, deadline - time.monotonic()))
            finally:
                try:
                    session.close()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
            if not size or size % 2:
                raise RuntimeError(f"翻译音频为空或不完整（{size} 字节）")
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
    if not args.language.strip():
        parser.error("--language 需要非空目标语种代码")
    if args.microphone and not 0.1 <= args.seconds <= 20:
        parser.error("--seconds 应在 0.1～20 之间")
    if not args.no_play:
        player_command(args.output)
    if args.wav:
        source = open_wav(args.wav, (16000,))
        try:
            duration = source.getnframes() / 16000
            if not 0.1 <= duration <= 20:
                parser.error("本示例的 WAV 输入限制为 0.1～20 秒")
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
        print("\n已取消；翻译文件可能不完整。", file=sys.stderr)
        sys.exit(130)
    except subprocess.SubprocessError as error:
        print(f"翻译音频已保存，但播放器失败：{error}", file=sys.stderr)
        sys.exit(1)
    except (DashScopeError, OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"翻译失败：{error}；已创建的输出文件可能不完整。", file=sys.stderr)
        sys.exit(1)
