# -*- coding: utf-8 -*-
"""HTTP TTS: synthesize once, download raw PCM, then save and play one WAV."""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from dashscope import Client
from ._common import (
    play_wav,
    player_command,
    positive_seconds,
    wav_output,
)


class SampleError(Exception):
    """Locally authored errors whose messages contain no signed URLs."""


def https_audio_url(url):
    """Validate the signed URL; change only an HTTP scheme to HTTPS."""
    if not isinstance(url, str) or not url:
        raise SampleError("服务未返回音频下载地址")
    if any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise SampleError("音频下载地址包含非法空白或控制字符")
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme in ("http", "https")
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and "#" not in url
            and "\\" not in url
        )
        # Validate a supplied port without changing the signed URL.
        _ = parts.port
    except ValueError:
        raise SampleError("音频下载地址无效") from None
    if not valid:
        raise SampleError("音频下载地址必须为 HTTP(S)，有主机且无 userinfo/fragment")
    if parts.scheme == "http":
        return "https" + url[len(parts.scheme):]
    return url


def make_parser():
    parser = argparse.ArgumentParser(
        description="HTTP 单次语音合成：下载原始 PCM，保存 WAV 后播放。真实请求可能计费。",
        epilog=(
            "在仓库根目录执行：python -m examples.01_atomic_capabilities.audio.direct_call.tts_http "
            "--text '你好，欢迎使用语音合成。' --output hello.wav\n"
            "使用 DASHSCOPE_API_KEY、DASHSCOPE_WORKSPACE 和模型对应地域。"
            "--no-play 仅保存文件。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="待合成的非空文本")
    source.add_argument("--text-file", type=Path, help="非空 UTF-8 文本文件")
    parser.add_argument(
        "--model", default="qwen-audio-3.1-tts-flash",
        choices=("qwen-audio-3.1-tts-flash", "qwen-audio-3.0-tts-flash",
                 "cosyvoice-v3-flash", "cosyvoice-v3-plus"),
    )
    parser.add_argument(
        "--voice", help="3.1 默认 longanhuan_v3.1；3.0 默认 longanhuan_v3.6；"
                        "CosyVoice 默认 longanhuan",
    )
    parser.add_argument("--sample-rate", type=int, default=24000,
                        choices=(8000, 16000, 22050, 24000, 44100, 48000))
    parser.add_argument("--output", type=Path, required=True,
                        help="新的 .wav 文件，不覆盖已有文件")
    parser.add_argument("--no-play", action="store_true", help="只保存，不播放")
    parser.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="合成及下载的网络超时秒数，默认 30")
    parser.add_argument("--region", help="覆盖 DASHSCOPE_REGION")
    parser.add_argument("--workspace", help="覆盖 DASHSCOPE_WORKSPACE")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    text = (args.text_file.read_text(encoding="utf-8")
            if args.text_file is not None else args.text)
    if not text.strip():
        parser.error("合成文本不能为空或仅含空白")
    voice = args.voice if args.voice is not None else {
        "qwen-audio-3.1-tts-flash": "longanhuan_v3.1",
        "qwen-audio-3.0-tts-flash": "longanhuan_v3.6",
        "cosyvoice-v3-flash": "longanhuan",
        "cosyvoice-v3-plus": "longanhuan",
    }[args.model]
    if not args.no_play:
        player_command(args.output)

    size = 0
    pending = b""
    header_checked = False
    # Reserve the output exclusively before making any billable request.
    raw, output = wav_output(args.output, args.sample_rate)
    try:
        client = Client(region=args.region, workspace=args.workspace,
                        timeout=args.timeout, max_retries=0)
        try:
            result = client.audio.tts.http.synthesize(
                text, model=args.model, voice=voice, format="pcm",
                sample_rate=args.sample_rate, timeout=args.timeout,
            )
            if result.is_error:
                raise SampleError("服务返回合成错误")
            print("request_id:", result.request_id, "usage:", result.usage)
            url = https_audio_url(result.audio_url)
            # No SDK credentials on downloads; redirects cannot downgrade HTTPS.
            downloader = httpx.Client(timeout=args.timeout, follow_redirects=False)
            try:
                request = downloader.build_request("GET", url)
                response = downloader.send(request, stream=True)
                try:
                    if not 200 <= response.status_code < 300:
                        raise SampleError(
                            f"音频下载失败（HTTP {response.status_code}；不跟随重定向）"
                        )
                    for data in response.iter_bytes():
                        if not data:
                            continue
                        if not header_checked:
                            pending += data
                            if len(pending) < 12:
                                continue
                            if pending[:4] == b"RIFF" and pending[8:12] == b"WAVE":
                                raise SampleError(
                                    "服务返回 RIFF/WAVE 容器而非原始 PCM；拒绝二次封装"
                                )
                            data, pending = pending, b""
                            header_checked = True
                        output.writeframesraw(data)
                        size += len(data)
                finally:
                    response.close()
            finally:
                downloader.close()
            # Very short PCM can legitimately be shorter than a WAV signature.
            if pending:
                output.writeframesraw(pending)
                size += len(pending)
            if not size or size % 2:
                raise SampleError(f"返回的 PCM 音频为空或不是偶数字节（{size} 字节）")
        finally:
            client.close()
    finally:
        try:
            output.close()
        finally:
            raw.close()
    print(f"已保存 {args.output.resolve()}（{size} 字节 PCM，"
          f"{size / 2 / args.sample_rate:.2f} 秒）")
    if not args.no_play:
        play_wav(args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消；已创建的输出文件可能不完整。", file=sys.stderr)
        sys.exit(130)
    except SampleError as error:
        print(f"合成失败：{error}；已创建的输出文件可能不完整。", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # SDK/httpx messages may contain signed URLs; do not print them.
        print(f"运行失败：{type(error).__name__}；已创建的输出文件可能不完整。"
              "播放失败时可使用 --no-play。", file=sys.stderr)
        sys.exit(1)
