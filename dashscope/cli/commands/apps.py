# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope apps`` -- Bailian application calls.

An application carries its own prompt, flow, plugins and knowledge base, so
the request has no ``model``: ``--app-id`` identifies what to call. What the
CLI adds on top is the streaming default -- the service sends cumulative text
unless asked otherwise, which would print the same sentence over and over, so
``--stream`` asks for incremental frames unless told not to.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import typer

from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_object_arg,
    parse_messages_arg,
    read_stdin,
    read_text_arg,
    stdin_is_piped,
)
from dashscope.cli.output import Renderer, render_text_stream

_HELP = r"""Call a Bailian application.

An application is addressed by `--app-id`; its own prompt, flow, plugins and
knowledge base decide the rest. Send one turn with `--prompt`, a transcript
you maintain with `--messages`, or let the service keep the state with
`--session-id` / `--memory-id`.

`--stream` writes incremental text to stdout in table mode, and one JSON
object per line in json/yaml mode.
"""

_COMPLETIONS_HELP = r"""Application completions."""


def _read_inputs(
    prompt: Optional[str],
    messages: Optional[str],
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Unlike chat, ``--prompt`` and ``--messages`` combine here.

    The SDK appends ``prompt`` as the latest user turn when both are given,
    so neither one shadows the other.
    """
    text = read_text_arg(prompt, option_name="--prompt")
    conversation = (
        parse_messages_arg(messages) if messages is not None else None
    )
    if text is None and conversation is None and stdin_is_piped():
        text = read_stdin().strip() or None
    if not text and not conversation:
        raise usage_error(
            "no input given: pass --prompt, --messages, or pipe text on stdin",
        )
    return text, conversation


def _render_completion(renderer: Renderer, result: Any) -> None:
    """Answer on stdout; thoughts, citations and usage on stderr.

    Citations go to stderr because they are diagnostics for a human, not
    part of the answer -- ``dashscope apps completions create ... > a.txt``
    should capture the text alone.
    """
    if not renderer.table_mode:
        renderer.print_object(result)
        return
    for thought in result.thoughts or ():
        if thought.thought:
            renderer.note(thought.thought)
    renderer.echo(result.text or "")
    for reference in result.doc_references or ():
        name = reference.doc_name or reference.title or reference.doc_id or "-"
        renderer.note(f"source: {name} {reference.doc_url or ''}".rstrip())
    if result.session_id:
        renderer.note(f"session_id: {result.session_id}")
    if result.usage is not None:
        renderer.note(f"usage: {result.usage.model_dump(exclude_none=True)}")


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    completions_app = typer.Typer(help=_COMPLETIONS_HELP, no_args_is_help=True)

    @completions_app.command("create")
    def completions_create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        app_id: str = typer.Option(
            ...,
            "--app-id",
            help="Application id, from the Bailian console.",
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
        stream: bool = typer.Option(
            False,
            "--stream",
            help="Stream the response instead of waiting for it to finish.",
        ),
        session_id: Optional[str] = typer.Option(
            None,
            "--session-id",
            help="Let the service keep the conversation state.",
        ),
        memory_id: Optional[str] = typer.Option(None, "--memory-id"),
        biz_params: Optional[str] = typer.Option(
            None,
            "--biz-params",
            help="Application flow and plugin inputs as a JSON object. "
            "Accepts @file.",
        ),
        image: Optional[List[str]] = typer.Option(
            None,
            "--image",
            help="Image URL for the application to read. Repeatable.",
        ),
        file: Optional[List[str]] = typer.Option(
            None,
            "--file",
            help="File URL for the application to read. Repeatable.",
        ),
        rag_options: Optional[str] = typer.Option(
            None,
            "--rag-options",
            help="Retrieval options as a JSON object. Accepts @file.",
        ),
        has_thoughts: Optional[bool] = typer.Option(
            None,
            "--has-thoughts/--no-has-thoughts",
            help="Ask for the planning and tool-call steps.",
        ),
        doc_tag_codes: Optional[List[str]] = typer.Option(
            None,
            "--doc-tag-code",
            help="Restrict retrieval to a tag. Repeatable.",
        ),
        doc_reference_type: Optional[str] = typer.Option(
            None,
            "--doc-reference-type",
            help="Citation style for retrieved documents.",
        ),
        incremental_output: Optional[bool] = typer.Option(
            None,
            "--incremental-output/--no-incremental-output",
            help="Stream deltas rather than cumulative text. Defaults to on "
            "with --stream.",
        ),
        temperature: Optional[float] = typer.Option(None, "--temperature"),
        top_p: Optional[float] = typer.Option(None, "--top-p"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        seed: Optional[int] = typer.Option(None, "--seed"),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object, passed through "
            "verbatim. Accepts @file.",
        ),
    ) -> None:
        """Send one application completion request."""
        obj = ctx.obj
        renderer = obj.renderer
        text, conversation = _read_inputs(prompt, messages)
        completions = obj.client.apps.completions
        kwargs: Dict[str, Any] = {
            "app_id": app_id,
            "prompt": text,
            "messages": conversation,
            "session_id": session_id,
            "memory_id": memory_id,
            "biz_params": parse_json_object_arg(
                biz_params,
                option_name="--biz-params",
            ),
            "image_list": list(image) if image else None,
            "file_list": list(file) if file else None,
            "rag_options": parse_json_object_arg(
                rag_options,
                option_name="--rag-options",
            ),
            "has_thoughts": has_thoughts,
            "doc_tag_codes": list(doc_tag_codes) if doc_tag_codes else None,
            "doc_reference_type": doc_reference_type,
            "incremental_output": (
                incremental_output
                if incremental_output is not None
                else (True if stream else None)
            ),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "parameters": parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        }

        if not stream:
            _render_completion(renderer, completions.create(**kwargs))
            return

        result = completions.create(stream=True, **kwargs)
        try:
            render_text_stream(renderer, result)
        finally:
            # Runs on Ctrl-C too, so the httpx stream is not left open.
            result.close()

    app.add_typer(completions_app, name="completions")
    return app
