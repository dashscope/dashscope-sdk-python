# -*- coding: utf-8 -*-
"""WebSocket duplex TTS: send text while receiving audio into one WAV."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dashscope import Client
from dashscope.v2.errors import DashScopeError
from ._common import (
    play_wav,
    player_command,
    positive_seconds,
    wav_output,
)


def receive_audio(session, output, events):
    size = 0
    for frame in session.events():
        if isinstance(frame, bytes):
            output.writeframesraw(frame)
            size += len(frame)
        elif events is not None:
            events.write(json.dumps(frame, ensure_ascii=False) + "\n")
            events.flush()
    return size


def synthesize(client, args, parts, output, events):
    options = {
        "model": args.model,
        "voice": args.voice,
        "format": "pcm",
        "sample_rate": args.sample_rate,
        "timeout": args.timeout,
        "enable_ssml": args.ssml,
    }
    if args.instruction:
        options["parameters"] = {"instruction": args.instruction}

    executor = None
    session = client.audio.tts.connect(**options)
    try:
        session.start()
        executor = ThreadPoolExecutor(max_workers=1)
        receiver = executor.submit(receive_audio, session, output, events)
        for part in parts:
            if receiver.done():
                receiver.result()
                raise RuntimeError("文本尚未发送完，合成会话已结束")
            if part:
                session.send_text(part)
        session.stop()
        size = receiver.result(timeout=args.timeout)
        print("request_id:", session.session.request_id)
        print("首包延迟(ms):", session.first_package_delay)
        print("usage:", session.usage)
        return size
    finally:
        try:
            # Close before executor shutdown to unblock its socket reader.
            session.close()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)


def make_parser():
    parser = argparse.ArgumentParser(
        description="WebSocket 双工流式合成：边发送文本边接收音频，保存 WAV 后播放。",
        epilog=(
            "默认 qwen-audio-3.1-tts-flash，推荐音色 longanhuan_v3.1。\n"
            "在仓库根目录执行：python -m examples.01_atomic_capabilities.audio.direct_call.tts_websocket "
            "--text '你好，欢迎使用语音合成。' --output hello.wav\n"
            "按行发送：python -m examples.01_atomic_capabilities.audio.direct_call.tts_websocket "
            "--text-file article.txt --output article.wav --events events.jsonl\n"
            "先安装 python -m pip install -e .，配置 DASHSCOPE_API_KEY、"
            "DASHSCOPE_WORKSPACE 和模型对应地域。真实服务可能计费。"
            "不调用独立 qwen-asr、旧 qwen-tts 或 Sambert。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="文本；可重复传入多个片段")
    source.add_argument("--text-file", type=Path, help="UTF-8 文本，按行发送")
    source.add_argument("--prompt", help="实时朗读大语言模型生成的回答，额外计费")
    parser.add_argument("--llm-model", default="qwen-plus", help="--prompt 的文本模型")
    parser.add_argument("--model", default="qwen-audio-3.1-tts-flash",
                        choices=("qwen-audio-3.1-tts-flash",
                                 "qwen-audio-3.0-tts-flash",
                                 "cosyvoice-v3-flash", "cosyvoice-v3-plus"))
    parser.add_argument("--voice",
                        help="3.1 默认 longanhuan_v3.1；3.0 默认 longanhuan_v3.6；"
                             "CosyVoice 默认 longanhuan；"
                             "复刻音色须同时指定创建时的 --model 和 --voice voice_id")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        choices=(8000, 16000, 22050, 24000, 44100, 48000))
    parser.add_argument("--output", type=Path, required=True,
                        help="新的 .wav 文件，不覆盖已有文件")
    parser.add_argument("--no-play", action="store_true", help="只保存，不播放")
    parser.add_argument("--events", type=Path, help="原始事件 JSONL 文件")
    parser.add_argument("--ssml", action="store_true", help="按所选模型的 SSML 规则合成")
    parser.add_argument("--instruction", help="模型支持时使用的合成指令")
    parser.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="socket 超时及输入结束后的等待秒数，默认 30")
    parser.add_argument("--region", help="覆盖 DASHSCOPE_REGION")
    parser.add_argument("--workspace", help="覆盖 DASHSCOPE_WORKSPACE")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.voice is None:
        args.voice = {
            "qwen-audio-3.1-tts-flash": "longanhuan_v3.1",
            "qwen-audio-3.0-tts-flash": "longanhuan_v3.6",
            "cosyvoice-v3-flash": "longanhuan",
            "cosyvoice-v3-plus": "longanhuan",
        }[args.model]
    if args.prompt is not None:
        if not args.prompt.strip():
            parser.error("--prompt 需要非空提示词")
        if args.ssml:
            parser.error("--prompt 输出为普通文本，不能与 --ssml 一起使用")
    if not args.no_play:
        player_command(args.output)
    source = args.text_file.open(encoding="utf-8") if args.text_file else None
    try:
        if source is not None:
            if not any(line.strip() for line in source):
                parser.error("文本文件不能为空")
            source.seek(0)
            parts = source
        else:
            parts = args.text or []
            if not args.prompt and not any(part.strip() for part in parts):
                parser.error("文本不能为空")
        raw, output = wav_output(args.output, args.sample_rate)
        try:
            events = args.events.open("x", encoding="utf-8") if args.events else None
            try:
                client = Client(
                    region=args.region, workspace=args.workspace, timeout=args.timeout,
                )
                try:
                    if args.prompt:
                        stream = client.chat.completions.create(
                            model=args.llm_model,
                            messages=[{"role": "user", "content": args.prompt}],
                            stream=True,
                            parameters={"incremental_output": True},
                        )
                        try:
                            size = synthesize(
                                client, args, stream.text_stream, output, events,
                            )
                        finally:
                            stream.close()
                    else:
                        size = synthesize(client, args, parts, output, events)
                    if not size or size % 2:
                        raise RuntimeError(f"返回的 PCM 音频为空或不完整（{size} 字节）")
                finally:
                    client.close()
            finally:
                if events is not None:
                    events.close()
        finally:
            try:
                output.close()
            finally:
                raw.close()
    finally:
        if source is not None:
            source.close()
    print(f"已保存 {args.output.resolve()}（{size / 2 / args.sample_rate:.2f} 秒）")
    if not args.no_play:
        play_wav(args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消；输出文件可能不完整。", file=sys.stderr)
        sys.exit(130)
    except subprocess.SubprocessError as error:
        print(f"音频已保存，但播放器失败：{error}", file=sys.stderr)
        sys.exit(1)
    except (DashScopeError, OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"合成失败：{error}；已创建的输出文件可能不完整。", file=sys.stderr)
        sys.exit(1)
