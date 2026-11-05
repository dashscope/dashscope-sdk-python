# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope audio`` -- speech synthesis, recognition and their registries.

Four subgroups: ``tts`` turns text into audio, ``asr`` turns audio into
text, ``realtime`` runs one turn over the duplex Omni/Qwen-Audio socket,
and the management plane (``vocabularies``, ``phrases``, ``voices``,
``voice-design``) holds the registries the first three consume.

``tts synthesize`` runs over websocket and hands back raw bytes, so the CLI
writes them to ``--out`` or to stdout -- ``dashscope audio tts synthesize
"hi" > hi.mp3``. ``tts stream`` writes each chunk as it arrives. ``tts http``
is the one-POST variant that answers with a URL instead of a payload, which
is what a pipeline that stores audio wants.

``asr transcribe`` is the batch service: submit file URLs and wait. ``asr
stream`` streams one local file over websocket and prints the sentences.

``realtime run`` is the one-turn slice of a socket that is otherwise
bidirectional and open-ended: it feeds a whole audio file, asks for one
response, prints the events and hangs up. A live microphone is not a shell
shape and stays in the SDK.

The management plane lives in :mod:`dashscope.cli.commands.
audio_customization` and is mounted by :func:`build_app` below.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import typer

from dashscope.cli.commands.audio_customization import build_admin_apps
from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_object_arg,
    read_text_arg,
    stdin_is_piped,
)
from dashscope.cli.output import Renderer
from dashscope.cli.progress import validate_wait_timeout, wait_job

_HELP = r"""Speech synthesis and recognition.

`audio tts synthesize` writes audio bytes to --out or stdout; `audio tts
stream` writes chunks as they arrive; `audio tts http` answers with a download
URL. `audio asr transcribe` submits file URLs to the batch service and waits;
`audio asr stream` streams one local file. `audio realtime run` feeds a file
to the duplex Omni socket and prints one turn. The `vocabularies`, `phrases`,
`voices` and `voice-design` subgroups manage the registries those commands
consume.
"""

_TTS_HELP = r"""Text to speech.

`synthesize` returns the complete audio; `stream` writes chunks as they
arrive. Both run over websocket and write to --out or stdout.
`http` is a single POST that answers with a URL.
"""

_ASR_HELP = r"""Speech to text.

`transcribe` submits public file URLs to the batch service and waits for the
transcripts. `stream` streams one local audio file over websocket and prints
the recognized sentences as they complete. `get` checks a batch task submitted
with --no-wait.
"""

_REALTIME_HELP = r"""One turn over the duplex realtime socket.

`run` opens a session, feeds a whole audio file, asks for one response and
hangs up -- the scriptable slice of a protocol that is otherwise an
open-ended conversation. Every server event goes out as one line, so
`-o json` gives an NDJSON trace of the turn.

A live microphone needs the SDK: `client.audio.realtime.connect()` stays
open across turns and expects audio at realtime pace.
"""

# Mirrors the SDK's own defaults so `--help` is self-documenting; the
# websocket resource takes them as non-optional keyword arguments.
DEFAULT_TTS_MODEL = "cosyvoice-v1"
DEFAULT_TTS_VOICE = "longxiaochun"
DEFAULT_TTS_FORMAT = "mp3"
DEFAULT_TTS_SAMPLE_RATE = 22050

DEFAULT_TRANSCRIBE_MODEL = "qwen-audio-3.1-asr-flash-filetrans"

DEFAULT_WAIT_TIMEOUT = 3600.0

_JOB_COLUMNS = ("task_id", "status", "submitted_at", "finished_at")

# The wire event that carries a recognized sentence.
_RESULT_EVENT = "result-generated"

# Same chunk size the SDK's own file-driven recognizer uses.
AUDIO_CHUNK_SIZE = 12800

# Realtime wire events, spelled as the service spells them.
_TEXT_DELTAS = ("response.text.delta", "response.audio_transcript.delta")
_AUDIO_DELTA = "response.audio.delta"
_TRANSCRIPT_DONE = ("conversation.item.input_audio_transcription.completed",)
_RESPONSE_DONE = "response.done"


def _read_audio(raw: Optional[str]) -> Optional[bytes]:
    """``--audio``: a path to raw audio, or ``-`` for stdin.

    There is no literal form here -- the value is always a source -- so
    ``speech.pcm`` and ``@speech.pcm`` mean the same thing.
    """
    if raw is None:
        return None
    if raw == "-":
        stream = getattr(sys.stdin, "buffer", None)
        if stream is None:  # pragma: no cover - stdin replaced by a text stub
            raise usage_error("cannot read audio from stdin: pass a path")
        return stream.read()
    path = Path(raw[1:] if raw.startswith("@") else raw).expanduser()
    if not path.is_file():
        raise usage_error(f"audio file not found: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise usage_error(f"cannot read {path}: {exc}") from exc


def _render_realtime_events(
    renderer: Renderer,
    events: Iterator[Dict[str, Any]],
    out: Optional[str],
) -> None:
    """Drain one realtime turn.

    Machine formats get every event verbatim as NDJSON: the trace *is* the
    result, and picking a subset here would hide the fields a caller came
    for. Table mode keeps stdout for the words -- the input transcript and
    the answer text -- and puts the spoken reply in ``--out`` only when one
    was named, because audio bytes and text cannot share a redirect.
    """
    audio = bytearray()
    for event in events:
        kind = event.get("type")
        if not renderer.table_mode:
            renderer.print_stream_chunk(event)
        elif kind in _TEXT_DELTAS:
            renderer.write(event.get("delta") or "")
        elif kind in _TRANSCRIPT_DONE:
            renderer.note(f"input: {event.get('transcript') or ''}")
        elif kind == _AUDIO_DELTA and out:
            audio.extend(base64.b64decode(event.get("delta") or ""))
        elif kind == _RESPONSE_DONE:
            response = event.get("response") or {}
            if response.get("usage"):
                renderer.note(f"usage: {response['usage']}")
        if kind == _RESPONSE_DONE:
            break
    if renderer.table_mode:
        renderer.echo()
    if out and audio:
        _write_audio(bytes(audio), out, renderer)


def _read_text(text: Optional[str], *, option_name: str) -> str:
    resolved = read_text_arg(text, option_name=option_name)
    if not resolved or not resolved.strip():
        raise usage_error(f"{option_name}: no text given")
    return resolved


def _write_audio(
    data: Union[bytes, Iterable[bytes]],
    out: Optional[str],
    renderer: Renderer,
) -> None:
    """Bytes to a file, or to stdout when no ``--out`` was given."""
    chunks = (data,) if isinstance(data, bytes) else data
    if out:
        path = Path(out).expanduser()
        try:
            output = path.open("wb")
        except OSError as exc:
            raise usage_error(f"cannot write {path}: {exc}") from exc
        size = 0
        with output:
            for chunk in chunks:
                try:
                    output.write(chunk)
                except OSError as exc:
                    raise usage_error(f"cannot write {path}: {exc}") from exc
                size += len(chunk)
        renderer.note(f"wrote {size} bytes to {path}")
        return
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:  # pragma: no cover - stdout replaced by a text stub
        raise usage_error(
            "cannot write audio to stdout: pass --out PATH instead",
        )
    for chunk in chunks:
        stream.write(chunk)
        stream.flush()


def _render_synthesis(renderer: Renderer, result: Any) -> None:
    if not renderer.table_mode:
        renderer.print_object(result)
        return
    url = result.audio_url
    if url:
        renderer.echo(url)
        return
    renderer.print_object(result)


def _sentences(events: Iterator[Dict[str, Any]]) -> Iterator[Tuple[Any, Dict]]:
    """Recognized sentences out of a raw websocket event stream.

    Heartbeats and empty frames are dropped: they carry no text and would
    print blank lines.
    """
    for event in events:
        header = event.get("header") or {}
        if header.get("event") != _RESULT_EVENT:
            continue
        payload = event.get("payload") or {}
        output = payload.get("output") or {}
        sentence = output.get("sentence") or {}
        if not sentence.get("text") or sentence.get("heartbeat"):
            continue
        yield event, sentence


def _render_job(renderer: Renderer, job: Any) -> None:
    """A finished transcription shows its transcript URLs."""
    if not renderer.table_mode:
        renderer.print_object(job)
        return
    urls = [
        result.transcription_url or result.url
        for result in job.results
        if result.transcription_url or result.url
    ]
    if urls:
        for url in urls:
            renderer.echo(url)
        return
    renderer.print_object(job, columns=_JOB_COLUMNS)


def _asr_jobs(client: Any) -> Any:
    from dashscope.resources._jobs import Jobs

    return Jobs(client, "audio/asr/transcription", "audio")


def build_app() -> typer.Typer:  # pylint: disable=too-many-statements
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    # ---- tts subgroup ---- #
    tts_app = typer.Typer(help=_TTS_HELP, no_args_is_help=True)

    @tts_app.command(
        "stream",
        help="Stream synthesized audio chunks as they arrive.",
    )
    @tts_app.command("synthesize")
    def tts_synthesize(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        text: str = typer.Argument(
            ...,
            help="Text to speak. Accepts @file and - for stdin.",
        ),
        model: str = typer.Option(DEFAULT_TTS_MODEL, "-m", "--model"),
        voice: str = typer.Option(DEFAULT_TTS_VOICE, "--voice"),
        audio_format: str = typer.Option(
            DEFAULT_TTS_FORMAT,
            "--format",
            help="Audio container: mp3, wav, pcm.",
        ),
        sample_rate: int = typer.Option(
            DEFAULT_TTS_SAMPLE_RATE,
            "--sample-rate",
        ),
        volume: Optional[int] = typer.Option(None, "--volume"),
        rate: Optional[float] = typer.Option(
            None,
            "--rate",
            help="Speaking rate multiplier.",
        ),
        pitch: Optional[float] = typer.Option(None, "--pitch"),
        language_hint: Optional[List[str]] = typer.Option(
            None,
            "--language-hint",
            help="Language hint, e.g. zh or en. Repeatable.",
        ),
        enable_ssml: Optional[bool] = typer.Option(
            None,
            "--enable-ssml/--no-enable-ssml",
            help="Parse the text as SSML.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
        out: Optional[str] = typer.Option(
            None,
            "-o",
            "--out",
            help="Where to write the audio. Defaults to stdout.",
        ),
    ) -> None:
        """Synthesize speech and write the audio out."""
        obj = ctx.obj
        tts = obj.client.audio.tts
        method = tts.stream if ctx.command.name == "stream" else tts.synthesize
        data = method(
            _read_text(text, option_name="TEXT"),
            model=model,
            voice=voice,
            format=audio_format,
            sample_rate=sample_rate,
            volume=volume,
            rate=rate,
            pitch=pitch,
            language_hints=list(language_hint) if language_hint else None,
            enable_ssml=enable_ssml,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        _write_audio(data, out, obj.renderer)

    @tts_app.command("http")
    def tts_http(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        text: str = typer.Argument(
            ...,
            help="Text to speak. Accepts @file and - for stdin.",
        ),
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Synthesis model, e.g. qwen3-tts-flash or cosyvoice-v2.",
        ),
        voice: str = typer.Option(..., "--voice"),
        audio_format: Optional[str] = typer.Option(
            None,
            "--format",
            help="Audio container. Defaults to the model's own.",
        ),
        sample_rate: Optional[int] = typer.Option(None, "--sample-rate"),
        language_type: Optional[str] = typer.Option(
            None,
            "--language-type",
            help="Dialect or language code, for the models that take one.",
        ),
        extra_input: Optional[str] = typer.Option(
            None,
            "--extra-input",
            help="Extra input fields as a JSON object. Accepts @file.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Synthesize over HTTP; prints the audio URL."""
        obj = ctx.obj
        result = obj.client.audio.tts.http.synthesize(
            _read_text(text, option_name="TEXT"),
            model=model,
            voice=voice,
            format=audio_format,
            sample_rate=sample_rate,
            language_type=language_type,
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        _render_synthesis(obj.renderer, result)

    # ---- asr subgroup ---- #
    asr_app = typer.Typer(help=_ASR_HELP, no_args_is_help=True)

    @asr_app.command("transcribe")
    def asr_transcribe(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        file_urls: List[str] = typer.Argument(
            ...,
            help="Public audio/video URLs to transcribe.",
        ),
        model: str = typer.Option(
            DEFAULT_TRANSCRIBE_MODEL,
            "-m",
            "--model",
        ),
        phrase_id: Optional[str] = typer.Option(
            None,
            "--phrase-id",
            help="Hot-word list to apply.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
        wait: bool = typer.Option(
            True,
            "--wait/--no-wait",
            help="Block until the transcripts are ready.",
        ),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="With --wait, how long to block before giving up (seconds).",
        ),
        submit_timeout: Optional[float] = typer.Option(
            None,
            "--submit-timeout",
            help="Timeout for the submit request only.",
        ),
    ) -> None:
        """Transcribe files with the batch service."""
        obj = ctx.obj
        renderer = obj.renderer
        kwargs = {
            "model": model,
            "phrase_id": phrase_id,
            "parameters": parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        }
        if not wait:
            job = obj.client.audio.asr.jobs.create(
                file_urls=list(file_urls),
                timeout=submit_timeout,
                **kwargs,
            )
            if renderer.table_mode:
                renderer.print_object(job, columns=_JOB_COLUMNS)
            else:
                renderer.print_object(job)
            return

        validate_wait_timeout(timeout)
        job = obj.client.audio.asr.transcribe(
            list(file_urls),
            timeout=timeout,
            submit_timeout=submit_timeout,
            **kwargs,
        )
        _render_job(renderer, job)

    @asr_app.command("get")
    def asr_get(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Transcription task id."),
    ) -> None:
        """Check a batch transcription task."""
        obj = ctx.obj
        renderer = obj.renderer
        job = obj.client.audio.asr.jobs.fetch(task_id)
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @asr_app.command("wait")
    def asr_wait(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Transcription task id."),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="How long to block before giving up (seconds).",
        ),
    ) -> None:
        """Block until a batch transcription task finishes."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        job = wait_job(
            obj.client,
            task_id,
            timeout=timeout,
            renderer=renderer,
            label="transcription task",
            waiter=obj.client.audio.asr.jobs,
        )
        _render_job(renderer, job)

    @asr_app.command("list")
    def asr_list(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Tasks per page."),
        status: Optional[str] = typer.Option(None, "--status"),
        model_name: Optional[str] = typer.Option(None, "--model"),
        api_key_id: Optional[str] = typer.Option(None, "--api-key-id"),
        region: Optional[str] = typer.Option(None, "--region"),
        start_time: Optional[str] = typer.Option(None, "--start-time"),
        end_time: Optional[str] = typer.Option(None, "--end-time"),
    ) -> None:
        """List batch transcription tasks."""
        obj = ctx.obj
        result = _asr_jobs(obj.client).list(
            page_no=page,
            page_size=size,
            status=status,
            model_name=model_name,
            api_key_id=api_key_id,
            region=region,
            start_time=start_time,
            end_time=end_time,
        )
        obj.renderer.print_rows(
            result.tasks,
            columns=_JOB_COLUMNS,
            envelope=result,
        )

    @asr_app.command("cancel")
    def asr_cancel(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Transcription task id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Cancel a batch transcription task."""
        confirm_or_die(f"cancel transcription task {task_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(_asr_jobs(obj.client).cancel(task_id))

    @asr_app.command("stream")
    def asr_stream(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        path: str = typer.Argument(..., help="Local audio file to recognize."),
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Recognition model, e.g. paraformer-realtime-v2.",
        ),
        audio_format: str = typer.Option(
            "pcm",
            "--format",
            help="Container of the input file: pcm, wav, mp3, opus, speex.",
        ),
        sample_rate: int = typer.Option(16000, "--sample-rate"),
        vocabulary_id: Optional[str] = typer.Option(
            None,
            "--vocabulary-id",
            help="Custom vocabulary to apply.",
        ),
        disfluency_removal: Optional[bool] = typer.Option(
            None,
            "--disfluency-removal/--no-disfluency-removal",
            help="Strip filler words from the transcript.",
        ),
        diarization: Optional[bool] = typer.Option(
            None,
            "--diarization/--no-diarization",
            help="Label who speaks each sentence.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Recognize one local audio file, printing each sentence."""
        obj = ctx.obj
        renderer = obj.renderer
        source = Path(path).expanduser()
        if not source.is_file():
            raise usage_error(f"audio file not found: {source}")
        events = obj.client.audio.asr.recognize_file(
            str(source),
            model=model,
            format=audio_format,
            sample_rate=sample_rate,
            vocabulary_id=vocabulary_id,
            disfluency_removal_enabled=disfluency_removal,
            diarization_enabled=diarization,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if not renderer.table_mode:
            for event in events:
                renderer.print_stream_chunk(event)
            return
        for _event, sentence in _sentences(events):
            # Interim results repeat the same growing sentence; only the
            # one carrying end_time is final.
            if sentence.get("end_time") is not None:
                renderer.echo(sentence["text"])

    # ---- realtime subgroup ---- #
    realtime_app = typer.Typer(help=_REALTIME_HELP, no_args_is_help=True)

    @realtime_app.command("run")
    def realtime_run(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Realtime model, e.g. qwen3.8-omni-flash-realtime.",
        ),
        audio: Optional[str] = typer.Option(
            None,
            "--audio",
            help="Raw audio to feed: a path, @path, or - for stdin. "
            "Defaults to piped stdin.",
        ),
        instructions: Optional[str] = typer.Option(
            None,
            "--instructions",
            help="Instructions for this one response. Accepts @file.",
        ),
        voice: Optional[str] = typer.Option(None, "--voice"),
        input_format: str = typer.Option(
            "pcm",
            "--input-format",
            help="Container of the audio being fed: pcm, wav.",
        ),
        output_format: Optional[str] = typer.Option(
            None,
            "--output-format",
            help="Container for the spoken reply, e.g. wav.",
        ),
        modality: Optional[List[str]] = typer.Option(
            None,
            "--modality",
            help="Output modality, e.g. text or audio. Repeatable.",
        ),
        transcribe_input: Optional[bool] = typer.Option(
            None,
            "--transcribe-input/--no-transcribe-input",
            help="Ask for a transcript of the audio being fed.",
        ),
        out: Optional[str] = typer.Option(
            None,
            "-o",
            "--out",
            help="Where to write the spoken reply. Without it the reply "
            "audio is dropped, so stdout stays readable.",
        ),
        timeout: Optional[float] = typer.Option(
            None,
            "--timeout",
            help="Socket timeout in seconds.",
        ),
    ) -> None:
        """Feed one audio file and print the turn that comes back."""
        obj = ctx.obj
        renderer = obj.renderer
        data = _read_audio(audio)
        if data is None and stdin_is_piped():
            data = _read_audio("-")
        if not data:
            raise usage_error(
                "no audio given: pass --audio PATH or pipe audio on stdin",
            )
        # The SDK's session options are sentinel-defaulted, so an unset
        # flag must stay absent from the call rather than arrive as None.
        settings: Dict[str, Any] = {"input_audio_format": input_format}
        if voice is not None:
            settings["voice"] = voice
        if output_format is not None:
            settings["output_audio_format"] = output_format
        if modality:
            settings["output_modalities"] = list(modality)
        if transcribe_input is not None:
            settings["enable_input_audio_transcription"] = transcribe_input
        prompt = read_text_arg(instructions, option_name="--instructions")
        session = obj.client.audio.realtime.connect(
            model=model,
            timeout=timeout,
        )
        with session:
            session.update_session(**settings)
            for start in range(0, len(data), AUDIO_CHUNK_SIZE):
                session.append_audio(data[start : start + AUDIO_CHUNK_SIZE])
            session.commit()
            if prompt:
                session.create_response(instructions=prompt)
            else:
                session.create_response()
            _render_realtime_events(renderer, session.events(), out)

    app.add_typer(tts_app, name="tts")
    app.add_typer(asr_app, name="asr")
    app.add_typer(realtime_app, name="realtime")
    for name, group in build_admin_apps().items():
        app.add_typer(group, name=name)
    return app
