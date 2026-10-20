# -*- coding: utf-8 -*-
"""WebSocket duplex ASR: send PCM from WAV while receiving sentences."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from ._common import open_wav, positive_seconds


class SampleError(RuntimeError):
    """A user-facing sample error with no transport or credential details."""


def add_recognition_arguments(parser):
    parser.add_argument(
        "--model", choices=("qwen-audio-3.1-asr-flash-streaming",
                            "paraformer-realtime-v2", "fun-asr-realtime"),
        default="qwen-audio-3.1-asr-flash-streaming",
        help="ASR model (default: qwen-audio-3.1-asr-flash-streaming); "
             "this example uses 16 kHz for Qwen-Audio and Fun-ASR",
    )
    parser.add_argument(
        "--transcript", type=Path, default=Path("transcript.txt"),
        help="new UTF-8 file for final sentences only (default: transcript.txt)",
    )
    parser.add_argument(
        "--partials", action="store_true",
        help="also display provisional text on stderr; never save it",
    )
    parser.add_argument(
        "--timeout", type=positive_seconds, default=30.0,
        help="socket timeout and final-result wait budget in seconds (default: 30)",
    )
    parser.add_argument("--region", help="DashScope region; default: SDK configuration")
    parser.add_argument("--workspace", help="DashScope workspace ID")
    parser.add_argument("--vocabulary-id", help="existing ASR vocabulary ID")


def check_receiver(receiver):
    if receiver.done():
        receiver.result()
        raise SampleError("ASR finished before all audio was sent.")


def consume_results(session, transcript, partials):
    for result in session.results():
        sentence = result.get_sentence()
        if not sentence or not sentence.get("text"):
            continue
        text = sentence["text"]
        if result.is_sentence_end():
            transcript.write(text + "\n")
            transcript.flush()
            print(text, flush=True)
        elif partials:
            print(f"[partial] {text}", file=sys.stderr, flush=True)


def run_recognition(args, sample_rate, transcript, send_audio):
    try:
        from dashscope import Client
    except ImportError:
        raise SampleError(
            "The V2 SDK is required; run python -m pip install -e . "
            "from the repository root."
        ) from None

    try:
        client = Client(
            timeout=args.timeout, region=args.region, workspace=args.workspace,
        )
        try:
            executor = None
            session = client.audio.asr.recognize(
                model=args.model,
                format="pcm",
                sample_rate=sample_rate,
                vocabulary_id=args.vocabulary_id,
                timeout=args.timeout,
            )
            try:
                executor = ThreadPoolExecutor(max_workers=1)
                receiver = None
                try:
                    session.start()
                    receiver = executor.submit(
                        consume_results, session, transcript, args.partials,
                    )
                    send_audio(session, receiver)
                    check_receiver(receiver)
                    session.stop()
                    try:
                        receiver.result(timeout=args.timeout)
                    except FutureTimeoutError:
                        if receiver.done():
                            receiver.result()
                        raise SampleError(
                            f"No task-finished within {args.timeout:g}s after stop."
                        ) from None
                except BaseException:
                    if receiver is not None and receiver.done():
                        receiver.result()
                    raise
            finally:
                try:
                    session.close()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
            print(json.dumps({
                "request_id": session.request_id,
                "first_package_delay_ms": session.first_package_delay,
                "last_package_delay_ms": session.last_package_delay,
                "usage": session.usage,
            }, ensure_ascii=False), file=sys.stderr, flush=True)
        finally:
            client.close()
    except SampleError:
        raise
    except Exception as error:
        # Transport exceptions can include authorization headers or response bodies.
        raise SampleError(
            f"ASR failed ({type(error).__name__}). Check network, SDK credentials, "
            "model/region access, audio device/output availability and --timeout. "
            "Output files may be incomplete."
        ) from None


def wav_frames(audio):
    remaining = audio.getnframes()
    block_frames = audio.getframerate() // 10
    while remaining:
        count = min(block_frames, remaining)
        pcm = audio.readframes(count)
        if len(pcm) != count * 2:
            raise SampleError("Truncated WAV: fewer PCM bytes than declared frames.")
        yield pcm
        remaining -= count
    if audio.readframes(1):
        raise SampleError("Invalid WAV: PCM data is not aligned to 16-bit frames.")


def validate_wav_data(audio):
    for _ in wav_frames(audio):
        pass
    audio.rewind()


def wait_until(deadline, receiver):
    check_receiver(receiver)
    delay = deadline - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    check_receiver(receiver)


def send_wav(audio, session, receiver):
    deadline = time.monotonic()
    for pcm in wav_frames(audio):
        wait_until(deadline, receiver)
        deadline = time.monotonic() + len(pcm) / (2 * audio.getframerate())
        session.send_audio_frame(pcm)
    wait_until(deadline, receiver)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "WebSocket duplex ASR with Qwen-Audio 3.1 streaming by default. "
            "Send a mono, 16-bit, 16 kHz PCM WAV while receiving text; not HTTP. "
            "WAV headers are removed and audio is paced in 100 ms frames. "
            "Uses existing SDK credentials. Final sentences go to stdout and "
            "the transcript; request ID, delays (ms) and usage go to stderr."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run from the repository root:\n"
            "  python -m pip install -e .\n"
            "  python -m examples.01_atomic_capabilities.audio.direct_call.asr_websocket speech.wav "
            "--transcript speech.txt\n"
            "  python -m examples.01_atomic_capabilities.audio.direct_call.asr_websocket speech.wav "
            "--model fun-asr-realtime --transcript fun.txt --partials\n\n"
            "Existing output files are never replaced. Failed runs may leave "
            "partial files.\n"
            "This example uses 16 kHz for Qwen-Audio and Fun-ASR; "
            "Paraformer also accepts 8 kHz. No resampling is performed."
        ),
    )
    parser.add_argument("file", type=Path, help="local uncompressed PCM WAV file")
    add_recognition_arguments(parser)
    args = parser.parse_args(argv)
    try:
        rates = (8000, 16000) if args.model == "paraformer-realtime-v2" else (16000,)
        audio = open_wav(args.file, sample_rates=rates)
        try:
            validate_wav_data(audio)
            transcript = args.transcript.open("x", encoding="utf-8")
            try:
                def send_audio(session, receiver):
                    send_wav(audio, session, receiver)

                run_recognition(args, audio.getframerate(), transcript, send_audio)
            finally:
                transcript.close()
        finally:
            audio.close()
        return 0
    except KeyboardInterrupt:
        print("error: interrupted; output may be incomplete.", file=sys.stderr)
        return 130
    except EOFError:
        print("error: invalid or truncated WAV header.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"error: {str(error) or type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
