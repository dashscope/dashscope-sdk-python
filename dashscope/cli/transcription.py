# -*- coding: utf-8 -*-
"""``transcription`` sub-command group."""
import json
from typing import List, Optional

import typer

import dashscope
from dashscope.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="transcription",
    help="Speech transcription commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Transcription request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    file_urls: List[str] = typer.Option(
        ...,
        "--file-url",
        help="Audio file URL, can be used multiple times",
    ),
    phrase_id: Optional[str] = typer.Option(
        None,
        "--phrase-id",
        help="Phrase id",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    channel_ids: Optional[List[int]] = typer.Option(
        None,
        "--channel-id",
        help="Selected channel id, can be used multiple times",
    ),
    disfluency_removal_enabled: bool = typer.Option(
        False,
        "--disfluency-removal-enabled",
        help="Whether to remove disfluency words",
    ),
    diarization_enabled: bool = typer.Option(
        False,
        "--diarization-enabled",
        help="Whether to enable speaker diarization",
    ),
    speaker_count: Optional[int] = typer.Option(
        None,
        "--speaker-count",
        help="Speaker count",
    ),
    timestamp_alignment_enabled: bool = typer.Option(
        False,
        "--timestamp-alignment-enabled",
        help="Whether to enable timestamp alignment",
    ),
    special_word_filter: Optional[str] = typer.Option(
        None,
        "--special-word-filter",
        help="Special word filter",
    ),
    audio_event_detection_enabled: bool = typer.Option(
        False,
        "--audio-event-detection-enabled",
        help="Whether to enable audio event detection",
    ),
):
    """Call speech transcription API."""
    response = dashscope.Transcription.call(
        model=model,
        file_urls=file_urls,
        phrase_id=phrase_id,
        workspace=workspace,
        channel_id=channel_ids,
        disfluency_removal_enabled=disfluency_removal_enabled,
        diarization_enabled=diarization_enabled,
        speaker_count=speaker_count,
        timestamp_alignment_enabled=timestamp_alignment_enabled,
        special_word_filter=special_word_filter,
        audio_event_detection_enabled=audio_event_detection_enabled,
    )
    # For async task creation, only check HTTP success, not business errors
    # Business errors will be reported when fetching/waiting for task results
    output = ensure_ok(response, check_business_error=False)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
