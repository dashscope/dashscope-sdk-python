# -*- coding: utf-8 -*-
"""Submit or resume URL-based batch transcription using the V2 speech API."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from dashscope import Client
from dashscope.v2.errors import DashScopeError
from ._common import positive_seconds


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
            "Use an HTTPS audio URL without userinfo or fragments, not a local "
            "path. Upload your own WAV to a service-reachable URL yourself; "
            "this sample never uploads files."
        )
    return value


def task_id(value):
    if not value or any(char.isspace() or char in "/\\?#" for char in value):
        raise argparse.ArgumentTypeError("task ID must be a nonempty ID, not a URL")
    return value


def checkpoint(output, data):
    output.seek(0)
    json.dump(data, output, ensure_ascii=False, indent=2)
    output.write("\n")
    output.truncate()
    output.flush()


def transcript_text(document):
    if (not isinstance(document, dict)
            or not isinstance(document.get("transcripts"), list)):
        raise ValueError("transcription JSON has no transcripts list")
    texts = []
    for transcript in document["transcripts"]:
        text = transcript.get("text")
        if text is None:
            text = "".join(sentence["text"] for sentence in transcript["sentences"])
        if not isinstance(text, str):
            raise ValueError("transcript text must be a string")
        texts.append(text)
    return "\n".join(texts)


def download_results(job, args, metadata, output):
    results = (job.output or {}).get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(
            "Task returned no per-file results; see task.json for its status"
        )
    failures = 0
    # Never reuse the SDK transport: signed download URLs need no API key.
    with httpx.Client(timeout=args.timeout, follow_redirects=True) as downloader:
        for index, result in enumerate(results, 1):
            record = {"index": index}
            try:
                if not isinstance(result, dict):
                    raise ValueError("invalid per-file result")
                record["subtask_status"] = result.get("subtask_status")
                if result.get("subtask_status") != "SUCCEEDED":
                    raise RuntimeError("subtask did not succeed")
                url = https_url(result.get("transcription_url") or "")
                response = downloader.get(url)
                response.raise_for_status()
                document = response.json()
                json_path = args.output_dir / f"{index:03d}.json"
                with json_path.open("x", encoding="utf-8") as artifact:
                    json.dump(document, artifact, ensure_ascii=False, indent=2)
                    artifact.write("\n")
                record["json"] = json_path.name
                text = transcript_text(document)
                text_path = args.output_dir / f"{index:03d}.txt"
                with text_path.open("x", encoding="utf-8") as artifact:
                    artifact.write(text + "\n")
                record["text"] = text_path.name
                print(f"File {index}: saved {json_path.name} and {text_path.name}",
                      flush=True)
            except (httpx.HTTPError, OSError, ValueError, RuntimeError,
                    argparse.ArgumentTypeError, KeyError, TypeError,
                    AttributeError) as error:
                failures += 1
                # Exception messages can contain presigned URLs; keep them out of logs.
                record["error"] = type(error).__name__
                if isinstance(error, httpx.HTTPStatusError):
                    record["http_status"] = error.response.status_code
                print(f"File {index}: FAILED ({record})", file=sys.stderr, flush=True)
            metadata["files"].append(record)
            checkpoint(output, metadata)
    return failures


def make_parser():
    parser = argparse.ArgumentParser(
        description="HTTP recorded-file ASR: submit a URL, poll, download transcripts. "
                    "Default: qwen-audio-3.1-asr-flash-filetrans; no WebSocket or SSE.",
        epilog=(
            "Upload your own WAV to an HTTPS URL reachable by the service first; "
            "local paths are NOT file_urls. This sample never uploads audio.\n"
            "Submit: python -m examples.01_atomic_capabilities.audio.asr_file --url \"$AUDIO_URL\" "
            "--output-dir batch-new\n"
            "Resume without a new paid submission: python -m examples.01_atomic_capabilities.audio.asr_file "
            "--task-id TASK_ID --output-dir batch-resumed\n"
            "Use your configured DASHSCOPE_API_KEY and the same region/workspace "
            "when resuming. Real service usage may incur charges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", action="append", type=https_url,
                        help="HTTPS audio URL; repeat for multiple files")
    source.add_argument("--task-id", type=task_id, help="resume an existing task")
    parser.add_argument("--model", default="qwen-audio-3.1-asr-flash-filetrans",
                        choices=("qwen-audio-3.1-asr-flash-filetrans",
                                 "fun-asr", "paraformer-v2"),
                        help="submission model (ignored when resuming)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory in an existing parent; never overwrite")
    parser.add_argument("--wait-timeout", type=positive_seconds, default=600.0,
                        help="maximum polling seconds (default: 600)")
    parser.add_argument("--timeout", type=positive_seconds, default=30.0,
                        help="SDK/download request timeout seconds (default: 30)")
    parser.add_argument("--region", help="override DASHSCOPE_REGION")
    parser.add_argument("--workspace", help="override DASHSCOPE_WORKSPACE")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if (args.url and args.model == "qwen-audio-3.1-asr-flash-filetrans"
            and len(args.url) != 1):
        parser.error("qwen-audio-3.1-asr-flash-filetrans requires exactly one --url")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir()  # Reserve the destination before any paid request.
    task_path = args.output_dir / "task.json"
    with task_path.open("x", encoding="utf-8") as output, Client(
        region=args.region, workspace=args.workspace, timeout=args.timeout,
        max_retries=0, raise_on_error=False,
    ) as client:
        metadata = {
            "model": None if args.task_id else args.model,
            "region": client.config.region,
            "workspace": client.config.workspace,
            "files": [],
        }
        if args.task_id:
            job = args.task_id
            metadata.update(task_id=args.task_id, status="RESUMING")
        else:
            print("Submitting a paid batch task; input URLs will not be logged.",
                  file=sys.stderr, flush=True)
            job = client.audio.asr.jobs.create(
                model=args.model, file_urls=args.url, parameters={},
            )
            if not job.task_id:
                raise RuntimeError(
                    f"No task ID returned (request_id={job.request_id}); "
                    "check the service before submitting again"
                )
            metadata.update(task_id=job.task_id, status=job.status)
        print(f"task_id: {metadata['task_id']}", flush=True)
        checkpoint(output, metadata)
        print("Saved task.json; resume with --task-id and a NEW --output-dir "
              "in the same region/workspace.", flush=True)
        job = client.audio.asr.jobs.wait(job, timeout=args.wait_timeout)
        metadata.update(status=job.status, request_id=job.request_id, code=job.code)
        checkpoint(output, metadata)
        failed = job.status != "SUCCEEDED" or job.is_error
        print(f"Task status: {job.status}", flush=True)
        # Even a failed batch may contain usable completed files.
        failures = download_results(job, args, metadata, output)
        if failed or failures:
            print(f"Batch unsuccessful; {failures} per-file failure(s). "
                  "Completed outputs are preserved.", file=sys.stderr)
            return 1
    print(f"Saved all transcripts in {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; keep task.json and resume, rather than resubmit.",
              file=sys.stderr)
        sys.exit(130)
    except (DashScopeError, httpx.HTTPError) as error:
        print(f"Request failed ({type(error).__name__}); keep task.json to resume. "
              "If submission returned no ID, check the service before retrying.",
              file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Batch failed: {error}. Existing outputs are preserved.",
              file=sys.stderr)
        sys.exit(1)
