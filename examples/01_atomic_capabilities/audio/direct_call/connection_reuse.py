# -*- coding: utf-8 -*-
"""Run five sequential ASR or TTS tasks on one real WebSocket connection."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Timer

from dashscope import Client
from ._common import open_wav, positive_seconds, wav_output
from .asr_websocket import (
    check_receiver,
    consume_results,
    send_wav,
    validate_wav_data,
)
from .tts_websocket import receive_audio


RUNS = 5


def completed_task(session):
    task_id = session.session.task_id
    header = (session.last_response or {}).get("header", {})
    if header.get("event") != "task-finished" or header.get("task_id") != task_id:
        raise RuntimeError("任务未收到匹配的 task-finished，停止复用连接")
    return {
        "task_id": task_id,
        "request_id": session.session.request_id,
        "final_event": header["event"],
        "first_package_delay_ms": session.first_package_delay,
        "usage": session.usage,
    }


def recognize_one(client, connection, args, source, index):
    source.rewind()
    path = args.output_dir / f"asr-{index:02d}.txt"
    transcript = path.open("x+", encoding="utf-8")
    try:
        session = client.audio.asr.recognize(
            model=args.model, format="pcm", sample_rate=16000,
            connection=connection, timeout=args.timeout,
        )
        executor = None
        try:
            session.start()
            print(f"ASR {index}/{RUNS}: task_id={session.session.task_id}", flush=True)
            executor = ThreadPoolExecutor(max_workers=1)
            receiver = executor.submit(
                consume_results, session, transcript, args.partials,
            )
            send_wav(source, session, receiver)
            check_receiver(receiver)
            session.stop()
            receiver.result(timeout=args.timeout)
            result = completed_task(session)
            result["last_package_delay_ms"] = session.last_package_delay
        finally:
            try:
                session.close()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
        transcript.seek(0)
        text = transcript.read().strip()
        if not text:
            raise RuntimeError("没有最终转写，请使用包含清晰语音的 WAV")
    finally:
        transcript.close()
    result.update(transcript=path.name, text=text)
    return result


def synthesize_one(client, connection, args, index):
    path = args.output_dir / f"tts-{index:02d}.wav"
    events_path = args.output_dir / f"tts-{index:02d}-events.jsonl"
    text = f"第{index}次。{args.text}"
    raw, output = wav_output(path, 24000)
    try:
        events = events_path.open("x", encoding="utf-8")
        try:
            session = client.audio.tts.connect(
                model=args.model, voice=args.voice, format="pcm", sample_rate=24000,
                connection=connection, timeout=args.timeout,
            )
            executor = None
            try:
                session.start()
                print(
                    f"TTS {index}/{RUNS}: task_id={session.session.task_id}",
                    flush=True,
                )
                executor = ThreadPoolExecutor(max_workers=1)
                receiver = executor.submit(receive_audio, session, output, events)
                session.send_text(text)
                session.stop()
                size = receiver.result(timeout=args.timeout)
                result = completed_task(session)
            finally:
                try:
                    session.close()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
        finally:
            events.close()
    finally:
        try:
            output.close()
        finally:
            raw.close()
    if not size or size % 2:
        raise RuntimeError(f"返回的 PCM 音频为空或不完整（{size} 字节）")
    result.update(audio=path.name, events=events_path.name, input_text=text,
                  audio_bytes=size, audio_seconds=size / 48000)
    return result


def checkpoint(output, data):
    output.seek(0)
    json.dump(data, output, ensure_ascii=False, indent=2)
    output.write("\n")
    output.truncate()
    output.flush()


def make_parser():
    parser = argparse.ArgumentParser(
        description="在一条真实 WebSocket 上顺序执行 5 次 ASR 或 TTS，不自动重连。",
        epilog=(
            "python -m examples.01_atomic_capabilities.audio.direct_call.connection_reuse asr "
            "tests/data/asr_example.wav --output-dir reuse-asr\n"
            "python -m examples.01_atomic_capabilities.audio.direct_call.connection_reuse tts "
            "--output-dir reuse-tts\n"
            "ASR 与 TTS 分别运行，各创建一条连接；不是混合并发测试。"
            "使用已配置的 SDK 凭据，真实调用可能计费。不会录音或播放。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True,
                        help="新的输出目录，父目录必须存在，不覆盖已有目录")
    common.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="握手、任务接收及结束输入后等待超时秒数，默认 30")
    common.add_argument("--task-timeout", type=positive_seconds, default=120.0,
                        help="每轮任务总预算秒数，超时关闭整条连接，默认 120")
    common.add_argument("--region", help="覆盖 DASHSCOPE_REGION")
    common.add_argument("--workspace", help="覆盖 DASHSCOPE_WORKSPACE")
    commands = parser.add_subparsers(dest="kind", required=True)
    asr = commands.add_parser("asr", parents=[common], help="同一 WAV 识别 5 次")
    asr.add_argument("file", type=Path, help="16 kHz、16-bit、单声道 PCM WAV")
    asr.add_argument("--model", default="qwen-audio-3.1-asr-flash-streaming",
                     choices=("qwen-audio-3.1-asr-flash-streaming",))
    asr.add_argument("--partials", action="store_true", help="显示临时识别文本")
    tts = commands.add_parser("tts", parents=[common], help="带轮次编号的文本合成 5 次")
    tts.add_argument("--model", default="qwen-audio-3.1-tts-flash",
                     choices=("qwen-audio-3.1-tts-flash",))
    tts.add_argument("--voice", default="longanhuan_v3.1", help="与 TTS 模型匹配的音色")
    tts.add_argument("--text", default="你好，这是连接复用测试。",
                     help="每轮自动在文本前加“第N次。”")
    return parser


def run_tasks(args, source=None):
    args.output_dir.mkdir()
    output = (args.output_dir / "summary.json").open("x", encoding="utf-8")
    try:
        data = {"kind": args.kind, "model": args.model, "runs_per_connection": RUNS,
                "status": "running", "connection_count": 0, "runs": []}
        checkpoint(output, data)
        try:
            client = Client(
                region=args.region, workspace=args.workspace,
                timeout=args.timeout, max_retries=0,
            )
            try:
                data.update(
                    region=client.config.region, workspace=client.config.workspace,
                )
                connection = client.ws_connection(timeout=args.timeout, module="audio")
                try:
                    data["connection_count"] = 1
                    checkpoint(output, data)
                    for index in range(1, RUNS + 1):
                        expired = Event()

                        def abort_task():
                            expired.set()
                            connection.close()

                        watchdog = Timer(args.task_timeout, abort_task)
                        watchdog.daemon = True
                        watchdog.start()
                        try:
                            if args.kind == "asr":
                                result = recognize_one(
                                    client, connection, args, source, index,
                                )
                            else:
                                result = synthesize_one(
                                    client, connection, args, index,
                                )
                        finally:
                            watchdog.cancel()
                            watchdog.join()
                        if expired.is_set():
                            raise TimeoutError("单轮任务超过 --task-timeout")
                        if connection.closed:
                            raise RuntimeError("共享连接已断开，停止后续任务")
                        if any(item["task_id"] == result["task_id"]
                               for item in data["runs"]):
                            raise RuntimeError("不同轮次使用了相同 task_id")
                        result.update(index=index, connection_open_after_task=True)
                        data["runs"].append(result)
                        checkpoint(output, data)
                        print(json.dumps(result, ensure_ascii=False), flush=True)
                finally:
                    connection.close()
                    data["connection_closed"] = connection.closed
            finally:
                client.close()
            data["status"] = "completed"
        except BaseException as error:
            data.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            checkpoint(output, data)
    finally:
        output.close()


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.kind == "asr":
        source = open_wav(args.file, (16000,))
        try:
            validate_wav_data(source)
            run_tasks(args, source)
        finally:
            source.close()
    else:
        if not args.text.strip():
            parser.error("--text 不能为空")
        run_tasks(args)
    print(f"完成：1 条连接，{RUNS} 个独立任务；{args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消并关闭连接；保留已完成结果。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"复用测试失败（{type(error).__name__}）；保留已有结果，不自动重试。",
              file=sys.stderr)
        sys.exit(1)
