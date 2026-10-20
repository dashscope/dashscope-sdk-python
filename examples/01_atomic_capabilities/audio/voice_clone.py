# -*- coding: utf-8 -*-
"""Create and manage Qwen-Audio and CosyVoice clones through the V2 public API."""

import argparse
import json
import re
import shlex
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from dashscope import Client
from dashscope.v2.errors import DashScopeError
from ._common import positive_seconds


MODELS = (
    "qwen-audio-3.0-tts-flash", "cosyvoice-v3-flash", "cosyvoice-v3-plus",
)


def https_url(value):
    try:
        url = urlsplit(value)
        valid = (
            url.scheme == "https" and url.hostname and url.port != 0
            and not url.username and not url.password and not url.fragment
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError(
            "Use a service-reachable HTTPS audio URL without userinfo or fragments, "
            "not a local path. Upload owned/authorized speaker audio yourself; "
            "this sample never uploads it."
        )
    return value


def voice_prefix(value):
    if not re.fullmatch(r"[a-z0-9]{1,10}", value):
        raise argparse.ArgumentTypeError("prefix must be 1-10 lowercase letters/digits")
    return value


def voice_id(value):
    if not value or any(char.isspace() for char in value):
        raise argparse.ArgumentTypeError(
            "voice ID must be nonempty and contain no spaces"
        )
    return value


def checkpoint(output, data, response):
    data["result"] = response.model_dump(mode="json", exclude={"headers"},
                                         exclude_none=True)
    if output is not None:
        output.seek(0)
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.truncate()
        output.flush()


def connection_options(client):
    options = []
    if client.config.region:
        options.extend(["--region", client.config.region])
    if client.config.workspace:
        options.extend(["--workspace", client.config.workspace])
    return options


def wait_ready(client, args, identifier, output, data):
    deadline = time.monotonic() + args.wait_timeout
    previous_status = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Voice is not ready before --wait-timeout; use get to recover"
            )
        voice = client.audio.voices.get(
            identifier, timeout=min(args.timeout, remaining)
        )
        checkpoint(output, data, voice)
        if voice.status != previous_status:
            print(f"Voice status: {voice.status}", flush=True)
            previous_status = voice.status
        if voice.status == "OK":
            return
        if voice.status != "DEPLOYING":
            raise RuntimeError(
                f"Voice is unavailable or has unsupported status: {voice.status}"
            )
        # Only DEPLOYING is retryable; never poll UNDEPLOYED or unknown states.
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2.0, remaining))


def create_voice(client, args, output, data):
    print("Use only your own or explicitly authorized speaker audio. "
          "Creating a clone may incur charges; it will NOT be automatically deleted.",
          file=sys.stderr, flush=True)
    voice = client.audio.voices.create(
        target_model=args.model, prefix=args.prefix, url=args.url,
    )
    identifier = voice.voice_id
    if not identifier:
        checkpoint(output, data, voice)
        raise RuntimeError(
            "No voice ID returned; use list before attempting another create"
        )
    print(f"voice_id: {identifier}", flush=True)
    data.update(voice_id=identifier, target_model=args.model)
    checkpoint(output, data, voice)
    print("Recover without creating again: " + shlex.join([
        "python", "-m", "examples.01_atomic_capabilities.audio.voice_clone", "get", "--voice-id", identifier,
        *connection_options(client),
    ]), flush=True)
    wait_ready(client, args, identifier, output, data)
    print("Ready. Synthesize separately (same model; may incur charges):")
    print(shlex.join([
        "python", "-m", "examples.01_atomic_capabilities.audio.tts_websocket", "--model", args.model,
        "--voice", identifier, "--text", "Hello, this is my cloned voice.",
        "--output", "cloned.wav", "--no-play", *connection_options(client),
    ]), flush=True)


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen-Audio / CosyVoice voice cloning; no synthesis, playback, "
            "or auto-deletion."
        ),
        epilog=(
            "Default target: qwen-audio-3.0-tts-flash (voice-enrollment API).\n"
            "Create: python -m examples.01_atomic_capabilities.audio.voice_clone create --url \"$AUDIO_URL\" "
            "--output new-voice.json\n"
            "Recover: python -m examples.01_atomic_capabilities.audio.voice_clone get --voice-id VOICE_ID\n"
            "List: python -m examples.01_atomic_capabilities.audio.voice_clone list --prefix demo\n"
            "Delete only when intended: python -m examples.01_atomic_capabilities.audio.voice_clone delete "
            "--voice-id VOICE_ID --yes\n"
            "Use your configured DASHSCOPE_API_KEY; keep the same region/workspace. "
            "Use subcommand --help for options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--output", type=Path, help="optional NEW .json file; never overwrite"
    )
    common.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="request timeout seconds (default: 30)")
    common.add_argument("--region", help="override DASHSCOPE_REGION")
    common.add_argument("--workspace", help="override DASHSCOPE_WORKSPACE")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", parents=[common], help="create a clone and wait until OK",
        description=(
            "Use only owned/authorized speaker audio at a service-reachable HTTPS URL. "
            "Upload it yourself; this sample never uploads local files. Recommended: "
            "10-20 seconds, WAV (16-bit)/MP3/M4A, >=16 kHz, <=10 MB, "
            "at most 60 seconds. "
            "Include at least 5 seconds of continuous clear speech, without music "
            "or other speakers. Only the first channel is used for stereo audio. "
            "The voice persists, including after timeout or interruption."
        ),
    )
    create.add_argument("--url", type=https_url, required=True)
    create.add_argument("--prefix", type=voice_prefix, default="demo",
                        help="1-10 lowercase letters/digits (default: demo)")
    create.add_argument("--model", choices=MODELS, default="qwen-audio-3.0-tts-flash",
                        help="target TTS model (default: qwen-audio-3.0-tts-flash); "
                             "synthesis must use this same version")
    create.add_argument("--wait-timeout", type=positive_seconds, default=300.0,
                        help="maximum readiness polling seconds (default: 300)")
    get = commands.add_parser("get", parents=[common],
                              help="query once; exit nonzero unless the voice is OK")
    get.add_argument("--voice-id", type=voice_id, required=True)
    listing = commands.add_parser(
        "list", parents=[common], help="list one page of clones"
    )
    listing.add_argument("--prefix", type=voice_prefix)
    listing.add_argument(
        "--page-index", type=int, default=0, help="zero-based page index"
    )
    listing.add_argument("--page-size", type=int, default=10)
    delete = commands.add_parser(
        "delete", parents=[common], help="explicitly delete one clone"
    )
    delete.add_argument("--voice-id", type=voice_id, required=True)
    delete.add_argument("--yes", action="store_true", help="confirm permanent deletion")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "delete" and not args.yes:
        parser.error("delete requires --yes; nothing has been deleted")
    if args.command == "list" and (args.page_index < 0 or args.page_size <= 0):
        parser.error("--page-index must be >= 0 and --page-size must be > 0")
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output must be a new .json file")
    # Open exclusively before any request can create a billable resource.
    output = args.output.open("x", encoding="utf-8") if args.output else None
    try:
        with Client(
            region=args.region, workspace=args.workspace, timeout=args.timeout,
            max_retries=0,
        ) as client:
            data = {"action": args.command, "region": client.config.region,
                    "workspace": client.config.workspace}
            if args.command == "create":
                create_voice(client, args, output, data)
            elif args.command == "get":
                voice = client.audio.voices.get(args.voice_id)
                data["voice_id"] = args.voice_id
                checkpoint(output, data, voice)
                print(f"voice_id: {args.voice_id}; status: {voice.status}", flush=True)
                if voice.status != "OK":
                    print(
                        "Voice is not ready; do not synthesize with it yet.",
                        file=sys.stderr,
                    )
                    return 1
            elif args.command == "list":
                voices = client.audio.voices.list(
                    prefix=args.prefix, page_index=args.page_index,
                    page_size=args.page_size,
                )
                data.update(page_index=args.page_index, page_size=args.page_size)
                checkpoint(output, data, voices)
                print(f"Page {args.page_index}: {len(voices.voices)} voice(s)")
                for voice in voices.voices:
                    print(
                        f"voice_id: {voice.get('voice_id')}; "
                        f"status: {voice.get('status')}"
                    )
            else:
                result = client.audio.voices.delete(args.voice_id)
                data["voice_id"] = args.voice_id
                checkpoint(output, data, result)
                print(f"Deleted {args.voice_id}; request_id: {result.request_id}")
    finally:
        if output is not None:
            output.close()
    if args.output:
        print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted; no automatic deletion. Recover with get/list, not create.",
            file=sys.stderr,
        )
        sys.exit(130)
    except (DashScopeError, httpx.HTTPError) as error:
        # Do not expose signed source/preview URLs through exception messages.
        print(f"Request failed ({type(error).__name__}). Use get/list to recover; "
              "existing clones and saved results are retained.", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, RuntimeError) as error:
        print(
            f"Voice operation failed: {error}. No automatic deletion.", file=sys.stderr
        )
        sys.exit(1)
