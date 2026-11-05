# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope audio`` management plane -- hot words, phrases and voices.

Four registries that the speech commands consume but do not own:

* ``vocabularies`` -- ASR hot-word tables, referenced by
  ``audio asr stream --vocabulary-id``;
* ``phrases`` -- compiled phrase sets, referenced by
  ``audio asr transcribe --phrase-id``;
* ``voices`` -- voices cloned from a sample recording;
* ``voice-design`` -- voices described in words instead of audio.

``voices`` and ``voice-design`` share one registry, one ``voice_id``
namespace and one quota, so ``voice-design list`` returns cloned voices
too. Both are mounted as subgroups of ``audio`` by
:func:`dashscope.cli.commands.audio.build_app`.

Pagination is not uniform across these endpoints and is exposed as the
service spells it rather than normalised away: the three customization
registries take a **0-based** ``--page-index``, while ``phrases`` is a
fine-tune-shaped endpoint and takes the 1-based ``--page``/``--size`` the
rest of the CLI uses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_text_arg,
)

_VOCABULARIES_HELP = "Manage ASR hot-word tables."
_PHRASES_HELP = "Manage compiled ASR phrase sets."
_VOICES_HELP = "Manage voices cloned from a sample recording."
_VOICE_DESIGN_HELP = "Manage voices described in words rather than audio."

_VOCABULARY_COLUMNS = (
    "vocabulary_id",
    "prefix",
    "target_model",
    "status",
)
_VOICE_COLUMNS = ("voice_id", "status", "target_model")
_PHRASE_COLUMNS = ("phrase_id", "status")


def _parse_entries(raw: Optional[str]) -> List[Dict[str, Any]]:
    """``--vocabulary``: a required JSON array of hot-word entries."""
    value = parse_json_arg(raw, option_name="--vocabulary")
    if value is None:
        raise usage_error("--vocabulary is required")
    if not isinstance(value, list):
        raise usage_error(
            f"--vocabulary: expected a JSON array, got {type(value).__name__}",
        )
    return value


def _parse_phrase_map(raw: Optional[str]) -> Dict[str, Any]:
    """``--phrases``: a required JSON object of phrase lists."""
    value = parse_json_object_arg(raw, option_name="--phrases")
    if value is None:
        raise usage_error("--phrases is required")
    if not value:
        raise usage_error("--phrases must not be empty")
    return value


def build_vocabularies_app() -> typer.Typer:
    """``audio vocabularies`` -- hot-word tables."""
    app = typer.Typer(help=_VOCABULARIES_HELP, no_args_is_help=True)

    @app.command("create")
    def create(
        ctx: typer.Context,
        target_model: str = typer.Option(
            ...,
            "--target-model",
            help="Recognizer the table is for, e.g. paraformer-v2.",
        ),
        prefix: str = typer.Option(
            ...,
            "--prefix",
            help="Becomes part of the generated vocabulary_id.",
        ),
        vocabulary: str = typer.Option(
            ...,
            "--vocabulary",
            help="Hot-word entries as a JSON array. Accepts @file.",
        ),
        model: Optional[str] = typer.Option(
            None,
            "-m",
            "--model",
            help="Customization model. Defaults to the service's own.",
        ),
    ) -> None:
        """Create a hot-word table."""
        obj = ctx.obj
        result = obj.client.audio.vocabularies.create(
            target_model=target_model,
            prefix=prefix,
            vocabulary=_parse_entries(vocabulary),
            model=model,
        )
        obj.renderer.print_object(result, columns=_VOCABULARY_COLUMNS)

    @app.command("list")
    def list_tables(
        ctx: typer.Context,
        prefix: Optional[str] = typer.Option(
            None,
            "--prefix",
            help="Only tables whose id starts with this prefix.",
        ),
        page_index: int = typer.Option(
            0,
            "--page-index",
            help="0-based page number.",
        ),
        page_size: int = typer.Option(
            10,
            "--page-size",
            help="Rows per page.",
        ),
    ) -> None:
        """List hot-word tables."""
        obj = ctx.obj
        result = obj.client.audio.vocabularies.list(
            prefix=prefix,
            page_index=page_index,
            page_size=page_size,
        )
        obj.renderer.print_rows(
            result.vocabularies,
            columns=_VOCABULARY_COLUMNS,
            envelope=result,
        )

    @app.command("get")
    def get(
        ctx: typer.Context,
        vocabulary_id: str = typer.Argument(..., help="Vocabulary id."),
    ) -> None:
        """Show one hot-word table and its contents."""
        obj = ctx.obj
        result = obj.client.audio.vocabularies.get(vocabulary_id)
        obj.renderer.print_object(result, columns=_VOCABULARY_COLUMNS)

    @app.command("update")
    def update(
        ctx: typer.Context,
        vocabulary_id: str = typer.Argument(..., help="Vocabulary id."),
        vocabulary: str = typer.Option(
            ...,
            "--vocabulary",
            help="Replacement entries as a JSON array. Accepts @file.",
        ),
    ) -> None:
        """Replace one table's contents."""
        obj = ctx.obj
        result = obj.client.audio.vocabularies.update(
            vocabulary_id,
            vocabulary=_parse_entries(vocabulary),
        )
        obj.renderer.print_object(result)

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        vocabulary_id: str = typer.Argument(..., help="Vocabulary id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a hot-word table."""
        confirm_or_die(f"delete vocabulary {vocabulary_id}?", yes=yes)
        obj = ctx.obj
        result = obj.client.audio.vocabularies.delete(vocabulary_id)
        obj.renderer.print_object(result)

    return app


def build_phrases_app() -> typer.Typer:
    """``audio phrases`` -- compiled phrase sets."""
    app = typer.Typer(help=_PHRASES_HELP, no_args_is_help=True)

    @app.command("create")
    def create(
        ctx: typer.Context,
        phrases: str = typer.Option(
            ...,
            "--phrases",
            help="Phrase lists as a JSON object. Accepts @file.",
        ),
        training_type: str = typer.Option(
            "compile_asr_phrase",
            "--training-type",
        ),
    ) -> None:
        """Compile a phrase set."""
        obj = ctx.obj
        result = obj.client.audio.phrases.create(
            phrases=_parse_phrase_map(phrases),
            training_type=training_type,
        )
        obj.renderer.print_object(result, columns=_PHRASE_COLUMNS)

    @app.command("list")
    def list_phrases(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Rows per page."),
    ) -> None:
        """List compiled phrase sets."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.audio.phrases.list(page_no=page, page_size=size)
        renderer.print_rows(
            result.phrases,
            columns=_PHRASE_COLUMNS,
            envelope=result,
        )
        if renderer.table_mode and result.total_count:
            pages = max(1, -(-result.total_count // max(1, result.page_size)))
            renderer.note(
                f"page {result.page_no}/{pages}, total {result.total_count}",
            )

    @app.command("get")
    def get(
        ctx: typer.Context,
        phrase_id: str = typer.Argument(..., help="Phrase set id."),
    ) -> None:
        """Show one compiled phrase set."""
        obj = ctx.obj
        result = obj.client.audio.phrases.get(phrase_id)
        obj.renderer.print_object(result, columns=_PHRASE_COLUMNS)

    @app.command("update")
    def update(
        ctx: typer.Context,
        phrase_id: str = typer.Argument(..., help="Phrase set id."),
        phrases: str = typer.Option(
            ...,
            "--phrases",
            help="Replacement phrase lists as a JSON object. Accepts @file.",
        ),
        training_type: str = typer.Option(
            "compile_asr_phrase",
            "--training-type",
        ),
    ) -> None:
        """Recompile a phrase set with new contents."""
        obj = ctx.obj
        result = obj.client.audio.phrases.update(
            phrase_id,
            phrases=_parse_phrase_map(phrases),
            training_type=training_type,
        )
        obj.renderer.print_object(result, columns=_PHRASE_COLUMNS)

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        phrase_id: str = typer.Argument(..., help="Phrase set id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a compiled phrase set."""
        confirm_or_die(f"delete phrase set {phrase_id}?", yes=yes)
        obj = ctx.obj
        result = obj.client.audio.phrases.delete(phrase_id)
        obj.renderer.print_object(result)

    return app


def build_voices_app() -> typer.Typer:
    """``audio voices`` -- voice cloning."""
    app = typer.Typer(help=_VOICES_HELP, no_args_is_help=True)

    @app.command("create")
    def create(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        target_model: str = typer.Option(
            ...,
            "--target-model",
            help="Synthesis model the voice is for, e.g. cosyvoice-v3-plus.",
        ),
        prefix: str = typer.Option(
            ...,
            "--prefix",
            help="Becomes part of the generated voice_id.",
        ),
        url: str = typer.Option(
            ...,
            "--url",
            help="Publicly readable sample recording to clone from.",
        ),
        language_hint: Optional[List[str]] = typer.Option(
            None,
            "--language-hint",
            help="Language of the sample, e.g. zh. Repeatable.",
        ),
        max_prompt_audio_length: Optional[float] = typer.Option(
            None,
            "--max-prompt-audio-length",
            help="Seconds of the sample to use.",
        ),
        extra_input: Optional[str] = typer.Option(
            None,
            "--extra-input",
            help="Extra input fields as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Clone a voice from a sample recording."""
        obj = ctx.obj
        result = obj.client.audio.voices.create(
            target_model=target_model,
            prefix=prefix,
            url=url,
            language_hints=(
                list(language_hint) if language_hint is not None else None
            ),
            max_prompt_audio_length=max_prompt_audio_length,
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
        )
        obj.renderer.print_object(result, columns=_VOICE_COLUMNS)

    @app.command("list")
    def list_voices(
        ctx: typer.Context,
        prefix: Optional[str] = typer.Option(
            None,
            "--prefix",
            help="Only voices whose id starts with this prefix.",
        ),
        page_index: int = typer.Option(
            0,
            "--page-index",
            help="0-based page number.",
        ),
        page_size: int = typer.Option(
            10,
            "--page-size",
            help="Rows per page.",
        ),
    ) -> None:
        """List cloned and designed voices (they share one registry)."""
        obj = ctx.obj
        result = obj.client.audio.voices.list(
            prefix=prefix,
            page_index=page_index,
            page_size=page_size,
        )
        obj.renderer.print_rows(
            result.voices,
            columns=_VOICE_COLUMNS,
            envelope=result,
        )

    @app.command("get")
    def get(
        ctx: typer.Context,
        voice_id: str = typer.Argument(..., help="Voice id."),
    ) -> None:
        """Show one cloned voice."""
        obj = ctx.obj
        result = obj.client.audio.voices.get(voice_id)
        obj.renderer.print_object(result, columns=_VOICE_COLUMNS)

    @app.command("update")
    def update(
        ctx: typer.Context,
        voice_id: str = typer.Argument(..., help="Voice id."),
        url: str = typer.Option(
            ...,
            "--url",
            help="Publicly readable replacement sample recording.",
        ),
    ) -> None:
        """Re-clone a voice from a different sample."""
        obj = ctx.obj
        result = obj.client.audio.voices.update(voice_id, url=url)
        obj.renderer.print_object(result)

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        voice_id: str = typer.Argument(..., help="Voice id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a cloned voice."""
        confirm_or_die(f"delete voice {voice_id}?", yes=yes)
        obj = ctx.obj
        result = obj.client.audio.voices.delete(voice_id)
        obj.renderer.print_object(result)

    return app


def build_voice_design_app() -> typer.Typer:
    """``audio voice-design`` -- voices described in words."""
    app = typer.Typer(help=_VOICE_DESIGN_HELP, no_args_is_help=True)

    @app.command("create")
    def create(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        target_model: str = typer.Option(
            ...,
            "--target-model",
            help="Synthesis model the voice is for.",
        ),
        prefix: str = typer.Option(
            ...,
            "--prefix",
            help="Becomes part of the generated voice_id.",
        ),
        voice_prompt: str = typer.Option(
            ...,
            "--voice-prompt",
            help="What the voice should sound like. Accepts @file and "
            "stdin (-).",
        ),
        preview_text: str = typer.Option(
            ...,
            "--preview-text",
            help="Text the preview is spoken with. Accepts @file and "
            "stdin (-).",
        ),
        language_hint: Optional[List[str]] = typer.Option(
            None,
            "--language-hint",
            help="Language of --preview-text. Only the first is honoured.",
        ),
        extra_input: Optional[str] = typer.Option(
            None,
            "--extra-input",
            help="Extra input fields as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Design a voice from a text prompt.

        The voice is unusable until its status leaves DEPLOYING; poll it
        with `audio voice-design get`.
        """
        obj = ctx.obj
        prompt = read_text_arg(voice_prompt, option_name="--voice-prompt")
        preview = read_text_arg(preview_text, option_name="--preview-text")
        if not prompt or not prompt.strip():
            raise usage_error("--voice-prompt must not be empty")
        if not preview or not preview.strip():
            raise usage_error("--preview-text must not be empty")
        result = obj.client.audio.voice_design.create(
            target_model=target_model,
            prefix=prefix,
            voice_prompt=prompt,
            preview_text=preview,
            language_hints=(
                list(language_hint) if language_hint is not None else None
            ),
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
        )
        obj.renderer.print_object(result, columns=_VOICE_COLUMNS)

    @app.command("list")
    def list_voices(
        ctx: typer.Context,
        prefix: Optional[str] = typer.Option(
            None,
            "--prefix",
            help="Only voices whose id starts with this prefix.",
        ),
        page_index: int = typer.Option(
            0,
            "--page-index",
            help="0-based page number.",
        ),
        page_size: int = typer.Option(
            10,
            "--page-size",
            help="Rows per page.",
        ),
    ) -> None:
        """List designed voices (cloned ones share the registry)."""
        obj = ctx.obj
        result = obj.client.audio.voice_design.list(
            prefix=prefix,
            page_index=page_index,
            page_size=page_size,
        )
        obj.renderer.print_rows(
            result.voices,
            columns=_VOICE_COLUMNS,
            envelope=result,
        )

    @app.command("get")
    def get(
        ctx: typer.Context,
        voice_id: str = typer.Argument(..., help="Voice id."),
    ) -> None:
        """Show one designed voice."""
        obj = ctx.obj
        result = obj.client.audio.voice_design.get(voice_id)
        obj.renderer.print_object(result, columns=_VOICE_COLUMNS)

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        voice_id: str = typer.Argument(..., help="Voice id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a designed voice."""
        confirm_or_die(f"delete voice {voice_id}?", yes=yes)
        obj = ctx.obj
        result = obj.client.audio.voice_design.delete(voice_id)
        obj.renderer.print_object(result)

    return app


def build_admin_apps() -> Dict[str, typer.Typer]:
    """The four management subgroups, keyed by their command name.

    Returned as a dict rather than mounted here so ``audio``'s own
    ``build_app`` stays the single place that decides the group's shape.
    """
    return {
        "vocabularies": build_vocabularies_app(),
        "phrases": build_phrases_app(),
        "voices": build_voices_app(),
        "voice-design": build_voice_design_app(),
    }
