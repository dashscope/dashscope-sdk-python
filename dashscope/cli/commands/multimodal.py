# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope multimodal`` -- the explicit multimodal-generation endpoint.

``dashscope chat create`` already routes a conversation that carries media
parts to this endpoint on its own, so these commands are for the two cases
that routing does not cover: a text-only conversation that still has to hit
the multimodal service (a qwen-vl model, say), and multimodal embeddings,
which have no text-only equivalent.

Message content parts are passed through untouched -- ``{"image": "a.png"}``
included -- because the SDK uploads local paths itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    parse_tools_arg,
    read_prompt_or_messages,
    read_text_arg,
)
from dashscope.cli.output import render_completion, render_text_stream

_HELP = r"""Call the multimodal-generation endpoint directly.

`multimodal chat create` is `chat create` against the multimodal service;
`multimodal embeddings create` embeds text, image, audio and video into one
shared vector space.
"""

_CHAT_HELP = r"""Multimodal chat completions.

Same conversation input as `dashscope chat create`: `--messages` (JSON or
JSONL, `@file` or `-` for stdin), `--prompt` with an optional `--system`, or
piped stdin. Content parts may name local paths; the SDK uploads them.
"""

_EMBEDDINGS_HELP = r"""Multimodal embeddings.

Each input item carries exactly one modality. Repeat `--text`, `--image`,
`--audio` or `--video` to embed several items in one call, or pass the whole
list as JSON with `--input` when the ordering across modalities matters.
"""

_MODALITIES = ("text", "image", "audio", "video")


def _parse_embedding_input(
    raw: Optional[str],
    modalities: Dict[str, Optional[List[str]]],
) -> List[Dict[str, Any]]:
    """Resolve the items to embed: ``--input`` wins over the flags."""
    if raw is not None:
        value = parse_json_arg(raw, option_name="--input")
        items = [value] if isinstance(value, dict) else value
        if not isinstance(items, list):
            raise usage_error(
                f"--input: expected a JSON object or array, got "
                f"{type(value).__name__}",
            )
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise usage_error(
                    f"--input: entry {index} is not a JSON object "
                    f"(got {type(item).__name__})",
                )
        if not items:
            raise usage_error("--input: no items to embed")
        return list(items)

    items = []
    for modality in _MODALITIES:
        for value in modalities.get(modality) or ():
            items.append({modality: value})
    if not items:
        raise usage_error(
            "nothing to embed: pass --text, --image, --audio, --video "
            "or --input",
        )
    return items


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    # ---- chat subgroup ---- #
    chat_app = typer.Typer(help=_CHAT_HELP, no_args_is_help=True)

    @chat_app.command("create")
    def chat_create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "-m",
            "--model",
            help="Model to call. Falls back to the configured default_model.",
        ),
        prompt: Optional[str] = typer.Option(
            None,
            "--prompt",
            help="Prompt text, @file, or - to read stdin.",
        ),
        messages: Optional[str] = typer.Option(
            None,
            "--messages",
            help="Conversation as a JSON array, JSONL, or an object with a "
            "messages key. Accepts @file and - for stdin.",
        ),
        system: Optional[str] = typer.Option(
            None,
            "--system",
            help="System prompt prepended to --prompt.",
        ),
        stream: bool = typer.Option(
            False,
            "--stream",
            help="Stream the response instead of waiting for it to finish.",
        ),
        tools: Optional[str] = typer.Option(
            None,
            "--tools",
            help="Tool definitions as a JSON array. Accepts @file.",
        ),
        temperature: Optional[float] = typer.Option(None, "--temperature"),
        top_p: Optional[float] = typer.Option(None, "--top-p"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
        stop: Optional[List[str]] = typer.Option(
            None,
            "--stop",
            help="Stop sequence. Repeatable.",
        ),
        seed: Optional[int] = typer.Option(None, "--seed"),
        enable_thinking: Optional[bool] = typer.Option(
            None,
            "--enable-thinking/--no-enable-thinking",
            help="Toggle reasoning output. Omit to use the model's default.",
        ),
        enable_search: Optional[bool] = typer.Option(
            None,
            "--enable-search/--no-enable-search",
            help="Toggle web search grounding. Omit to use the default.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra request parameters as a JSON object, passed through "
            "verbatim. Accepts @file.",
        ),
    ) -> None:
        """Send one multimodal chat completion request."""
        obj = ctx.obj
        renderer = obj.renderer
        kwargs: Dict[str, Any] = {
            "model": obj.resolve_model(model),
            "messages": read_prompt_or_messages(
                prompt=prompt,
                messages=messages,
                system=system,
            ),
            "tools": parse_tools_arg(tools),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "stop": list(stop) if stop else None,
            "seed": seed,
            "enable_thinking": enable_thinking,
            "enable_search": enable_search,
            "parameters": parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        }
        chat = obj.client.multimodal.chat

        if not stream:
            render_completion(renderer, chat.create(**kwargs))
            return

        result = chat.create(stream=True, **kwargs)
        try:
            render_text_stream(renderer, result)
        finally:
            # Runs on Ctrl-C too, so the httpx stream is not left open.
            result.close()

    # ---- embeddings subgroup ---- #
    embeddings_app = typer.Typer(help=_EMBEDDINGS_HELP, no_args_is_help=True)

    @embeddings_app.command("create")
    def embeddings_create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Multimodal embedding model, e.g. multimodal-embedding-v1.",
        ),
        text: Optional[List[str]] = typer.Option(
            None,
            "--text",
            help="Text to embed. Repeatable. Accepts @file.",
        ),
        image: Optional[List[str]] = typer.Option(
            None,
            "--image",
            help="Image URL or local path. Repeatable.",
        ),
        audio: Optional[List[str]] = typer.Option(
            None,
            "--audio",
            help="Audio URL or local path. Repeatable.",
        ),
        video: Optional[List[str]] = typer.Option(
            None,
            "--video",
            help="Video URL or local path. Repeatable.",
        ),
        input_json: Optional[str] = typer.Option(
            None,
            "--input",
            help="Items as a JSON array (or one object), each setting a "
            "single modality key. Overrides the per-modality flags. "
            "Accepts @file and - for stdin.",
        ),
        dimension: Optional[int] = typer.Option(
            None,
            "--dimension",
            help="Output vector dimension.",
        ),
        output_type: Optional[str] = typer.Option(None, "--output-type"),
        fps: Optional[float] = typer.Option(
            None,
            "--fps",
            help="Video sampling rate (snapshot models only).",
        ),
        instruct: Optional[str] = typer.Option(
            None,
            "--instruct",
            help="Instruction prefix for the embedding model.",
        ),
        enable_fusion: Optional[bool] = typer.Option(
            None,
            "--enable-fusion/--no-enable-fusion",
            help="Fuse the items into one vector instead of one per item.",
        ),
        res_level: Optional[int] = typer.Option(
            None,
            "--res-level",
            help="Resolution tier, 0-3 (snapshot models only).",
        ),
        max_video_frames: Optional[int] = typer.Option(
            None,
            "--max-video-frames",
            help="Frame cap (snapshot models only).",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Embed text, image, audio and video into one vector space."""
        obj = ctx.obj
        renderer = obj.renderer
        # --text accepts @file like every other payload option, so each
        # value goes through the reader rather than being taken literally.
        resolved = {
            modality: [
                read_text_arg(value, option_name=f"--{modality}")
                for value in values
            ]
            for modality, values in (
                ("text", text),
                ("image", image),
                ("audio", audio),
                ("video", video),
            )
            if values
        }
        result = obj.client.multimodal.embeddings.create(
            model=model,
            input=_parse_embedding_input(input_json, resolved),
            dimension=dimension,
            output_type=output_type,
            fps=fps,
            instruct=instruct,
            enable_fusion=enable_fusion,
            res_level=res_level,
            max_video_frames=max_video_frames,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if not renderer.table_mode:
            renderer.print_object(result)
            return
        count = len(result.embeddings)
        dims = (
            len(result.embeddings[0].embedding)
            if count and result.embeddings[0].embedding
            else 0
        )
        renderer.echo(f"{count} embedding(s), {dims} dimensions")
        if result.usage:
            renderer.note(
                f"usage: {result.usage.model_dump(exclude_none=True)}",
            )

    app.add_typer(chat_app, name="chat")
    app.add_typer(embeddings_app, name="embeddings")
    return app
