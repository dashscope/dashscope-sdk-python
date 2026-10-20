# -*- coding: utf-8 -*-
"""REST ASR: submit complete WAV audio once and receive cumulative text via SSE."""

import argparse
import base64
import json
import sys
from pathlib import Path

from httpx_sse import EventSource

from dashscope import Client
from dashscope.v2.errors import DashScopeError
from ._common import open_wav, positive_seconds
from .asr_file import https_url


PATH = "/services/aigc/multimodal-generation/generation"


def make_parser():
    parser = argparse.ArgumentParser(
        description="RESTful ASR：一次提交完整 WAV，以 HTTP/SSE 接收累计转写。",
        epilog=(
            "在仓库根目录执行：python -m examples.01_atomic_capabilities.audio.direct_call.asr_http_stream "
            "--file tests/data/asr_example.wav --transcript rest.txt\n"
            "本地 WAV 以 Base64 随请求发送，或使用 --url 传服务可访问的 HTTPS 地址。"
            "这不是实时音频输入；官方仅对至少 60 秒的音频分批返回。真实服务可能计费。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="本地单声道 16-bit PCM WAV")
    source.add_argument("--url", type=https_url, help="服务可下载的 HTTPS WAV URL")
    parser.add_argument("--model", default="qwen-audio-3.1-asr-flash",
                        choices=("qwen-audio-3.1-asr-flash",
                                 "qwen-audio-3.0-asr-flash"))
    parser.add_argument("--sample-rate", type=int, default=16000, choices=(16000,),
                        help="此示例只发送 16 kHz WAV，不自动重采样")
    parser.add_argument("--transcript", type=Path, required=True,
                        help="新的 UTF-8 TXT；每个累计结果覆盖本次文件内容")
    parser.add_argument("--events", type=Path, help="可选：保存服务端原始 JSONL 事件")
    parser.add_argument("--timeout", type=positive_seconds, default=120.0,
                        help="网络超时秒数；长音频首个 SSE 事件可能较慢，默认 120")
    parser.add_argument("--region", help="覆盖 DASHSCOPE_REGION")
    parser.add_argument("--workspace", help="覆盖 DASHSCOPE_WORKSPACE")
    return parser


def receive_transcript(response, transcript, events=None):
    text = ""
    count = 0
    request_id = response.headers.get("x-request-id")
    usage = None
    for event in EventSource(response).iter_sse():
        if not event.data:
            continue
        body = json.loads(event.data)
        if not isinstance(body, dict):
            raise ValueError("ASR SSE 事件必须为 JSON 对象")
        request_id = body.get("request_id") or request_id
        if event.event == "error" or body.get("code"):
            raise DashScopeError("ASR SSE 返回业务错误", request_id=request_id,
                                 code=body.get("code"))
        if events is not None:
            events.write(json.dumps(body, ensure_ascii=False) + "\n")
            events.flush()
        if body.get("usage") is not None:
            usage = body["usage"]
        output = body.get("output")
        if not isinstance(output, dict) or "text" not in output:
            continue
        text = output["text"]
        if not isinstance(text, str):
            raise ValueError("ASR output.text 必须为字符串")
        count += 1
        # output.text is cumulative and may revise an earlier partial result.
        transcript.seek(0)
        transcript.write(text)
        transcript.truncate()
        transcript.flush()
        print(f"[累计结果 {count}] {text}", flush=True)
    if not count:
        raise RuntimeError("HTTP 流结束，但没有收到 output.text")
    print(json.dumps({
        "request_id": request_id, "text_events": count, "usage": usage,
    }, ensure_ascii=False), file=sys.stderr, flush=True)
    return text


def main(argv=None):
    args = make_parser().parse_args(argv)
    audio_data = args.url
    if args.file is not None:
        audio = open_wav(args.file, sample_rates=(args.sample_rate,))
        try:
            pcm = audio.readframes(audio.getnframes())
            if len(pcm) != audio.getnframes() * 2 or audio.readframes(1):
                raise ValueError("WAV 的 PCM 数据长度与文件头不一致")
        finally:
            audio.close()
        audio_data = "data:audio/wav;base64," + base64.b64encode(
            args.file.read_bytes(),
        ).decode("ascii")
    body = {
        "model": args.model,
        "input": {"messages": [{
            "role": "user",
            "content": [{"type": "input_audio",
                         "input_audio": {"data": audio_data}}],
        }]},
        "parameters": {"format": "wav", "sample_rate": str(args.sample_rate)},
    }
    transcript = args.transcript.open("x+", encoding="utf-8")
    try:
        events = args.events.open("x", encoding="utf-8") if args.events else None
        try:
            client = Client(
                region=args.region, workspace=args.workspace,
                timeout=args.timeout, max_retries=0,
            )
            try:
                # Generic chat normalization would change this ASR-specific schema.
                response = client.transport.open_stream(
                    "POST", PATH, json=body, headers={"X-DashScope-SSE": "enable"},
                    timeout=args.timeout, sdk_module="audio",
                )
                try:
                    receive_transcript(response, transcript, events)
                finally:
                    response.close()
            finally:
                client.close()
        finally:
            if events is not None:
                events.close()
    finally:
        transcript.close()
    print(f"HTTP 流已结束，最后累计文本已保存到 {args.transcript.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消；输出可能是未完成的累计结果。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"REST ASR 失败（{type(error).__name__}，"
              f"request_id={getattr(error, 'request_id', None)}）；"
              "保留已有结果，请检查输入、网络、模型和地域。", file=sys.stderr)
        sys.exit(1)
