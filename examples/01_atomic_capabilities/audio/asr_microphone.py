# -*- coding: utf-8 -*-
"""Capture a system microphone and stream PCM to V2 ASR in the main thread."""

import argparse
import math
import signal
import sys
import time
from pathlib import Path

from ._common import (
    input_device,
    load_sounddevice,
    microphone,
    positive_seconds,
    wav_output,
)
from .asr_websocket import (
    SampleError,
    add_recognition_arguments,
    check_receiver,
    run_recognition,
)


def send_microphone(stream, recording, args, frame_count, session, receiver):
    interrupted = False

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    # Raising during an SDK send would close the socket before results can drain.
    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        stream.start()
        deadline = time.monotonic() + args.seconds
        print(
            f"Recording up to {args.seconds:g}s; Ctrl+C stops capture and drains ASR.",
            file=sys.stderr, flush=True,
        )
        remaining = frame_count
        while remaining and not interrupted and time.monotonic() < deadline:
            check_receiver(receiver)
            count = min(args.sample_rate // 10, remaining)
            data, overflowed = stream.read(count)
            check_receiver(receiver)
            if overflowed:
                raise SampleError(
                    "Microphone input overflowed; audio was lost. "
                    "Reduce system load or check the network before retrying."
                )
            pcm = bytes(data)
            if len(pcm) != count * 2:
                raise SampleError("Microphone returned an incomplete PCM frame.")
            if recording is not None:
                recording.writeframesraw(pcm)
            session.send_audio_frame(pcm)
            remaining -= count
    finally:
        try:
            stream.stop()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
    if interrupted:
        print("Capture interrupted; draining final ASR results...", file=sys.stderr)


def recognize_microphone(args, transcript, recording, frame_count):
    with microphone(args.sample_rate, device=args.device) as stream:
        # Preflight input before connecting, then avoid buffering during setup.
        stream.stop()

        def send_audio(session, receiver):
            send_microphone(
                stream, recording, args, frame_count, session, receiver,
            )

        run_recognition(args, args.sample_rate, transcript, send_audio)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Recognize system-microphone audio with Qwen-Audio 3.1 streaming ASR "
            "by default. Capture uses blocking 100 ms mono int16 reads, "
            "never network calls in an audio callback. "
            "Uses existing SDK credentials. Ctrl+C during capture stops input, "
            "then waits for final sentences. Final text goes to stdout and the "
            "transcript; request ID, delays (ms) and usage go to stderr."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run from the repository root:\n"
            "  python -m pip install -e '.[acli-voice]'\n"
            "  python -m examples.01_atomic_capabilities.audio.asr_microphone --list-devices\n"
            "  python -m examples.01_atomic_capabilities.audio.asr_microphone --seconds 10 "
            "--transcript microphone.txt\n"
            "  python -m examples.01_atomic_capabilities.audio.asr_microphone --device 1 --seconds 5 "
            "--recording take.wav --transcript take.txt --partials\n"
            "  python -m examples.01_atomic_capabilities.audio.asr_microphone --model fun-asr-realtime "
            "--no-recording --transcript fun.txt\n\n"
            "PortAudio and OS microphone permission are required for recording. "
            "Linux may need a system PortAudio package.\n"
            "--list-devices is offline: no Client, recording, or output files.\n"
            "Existing outputs are never replaced; failed runs may leave partial "
            "files. No resampling is performed."
        ),
    )
    add_recognition_arguments(parser)
    parser.add_argument(
        "--seconds", type=positive_seconds, default=10.0,
        help="maximum capture duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--sample-rate", type=int, choices=(8000, 16000), default=16000,
        help="microphone PCM sample rate (default: 16000); "
             "8000 is only enabled for Paraformer in this example",
    )
    parser.add_argument(
        "--device", type=input_device,
        help="sounddevice input index or name substring (default: system input)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="list local audio devices and exit without connecting or recording",
    )
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument(
        "--recording", type=Path, default=Path("recording.wav"),
        help="new WAV file for captured PCM (default: recording.wav)",
    )
    outputs.add_argument(
        "--no-recording", dest="recording", action="store_const", const=None,
        help="do not save the captured audio",
    )
    args = parser.parse_args(argv)
    try:
        if args.list_devices:
            print(load_sounddevice().query_devices())
            return 0
        if args.model != "paraformer-realtime-v2" and args.sample_rate != 16000:
            raise SampleError(
                f"This example uses --sample-rate 16000 for {args.model}."
            )
        frame_count = max(1, math.ceil(args.seconds * args.sample_rate))
        load_sounddevice()
        with args.transcript.open("x", encoding="utf-8") as transcript:
            if args.recording is not None:
                with wav_output(args.recording, args.sample_rate) as recording:
                    recognize_microphone(args, transcript, recording, frame_count)
            else:
                recognize_microphone(args, transcript, None, frame_count)
        return 0
    except KeyboardInterrupt:
        print("error: interrupted; output may be incomplete.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {str(error) or type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
