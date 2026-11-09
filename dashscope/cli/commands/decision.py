# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope decision`` -- the System One decision model.

One forward pass answers a set of *typed* questions about a business
``state``: multi-choice (``choice``), yes/no (``noul``) and ordinal scale
(``score``), each with its probability distribution and -- for ``choice``
and ``score`` -- a confidence. The model generates no text, so this is
the cheap way to route a ticket, moderate a payload or verify an agent
result from a shell script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_text_arg,
)

_HELP = r"""Answer typed questions about a business state.

`decision create` takes a `--state` (free text, a dialogue or a JSON
object) and a `--questions` JSON object whose values declare their own
`type`:

    {"department": {"type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "...", "technical": "..."}},
     "escalate":   {"type": "noul", "instructions": "Page on-call?"},
     "severity":   {"type": "score", "criteria": ["low", "mid", "high"]}}

The answer keys mirror the question ids. `choice` answers name an option,
`noul` answers are P(yes), and `score` answers are a probability-weighted
level index that may fall between two levels.
"""

_COLUMNS = ("question", "type", "answer", "confidence")


def _parse_state(raw: Optional[str]) -> Any:
    """``--state``: a JSON object/array, or a literal string.

    A ticket body is usually plain text and a routing payload is usually
    structured, so the same option takes both -- decided by the first
    character, the way ``embeddings create --input`` decides.
    """
    text = read_text_arg(raw, option_name="--state")
    if text is None:
        return None
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        value = parse_json_arg(raw, option_name="--state")
        if not isinstance(value, (dict, list)):
            raise usage_error(
                "--state: expected a JSON object or array, got "
                f"{type(value).__name__}",
            )
        return value
    return text


def _answer_value(answer: Any) -> Any:
    """Whichever of the three typed fields the answer actually filled."""
    for field in ("choice", "noul", "score"):
        value = getattr(answer, field, None)
        if value is not None:
            return value
    return None


def _rows(result: Any) -> List[Dict[str, Any]]:
    return [
        {
            "question": name,
            "type": getattr(answer, "type", None),
            "answer": _answer_value(answer),
            "confidence": getattr(answer, "confidence", None),
        }
        for name, answer in (result.answers or {}).items()
    ]


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("create")
    def create(
        ctx: typer.Context,
        model: str = typer.Option(
            "decision-model-preview",
            "-m",
            "--model",
            help="Decision model name.",
        ),
        state: Optional[str] = typer.Option(
            None,
            "--state",
            help="Business state under decision: text, or a JSON "
            "object/array. Accepts @file and stdin (-).",
        ),
        questions: str = typer.Option(
            ...,
            "--questions",
            help="Questions as a JSON object keyed by question id. "
            "Accepts @file and stdin (-).",
        ),
    ) -> None:
        """Run one decision round."""
        obj = ctx.obj
        renderer = obj.renderer
        parsed_state = _parse_state(state)
        if parsed_state is None:
            raise usage_error("--state is required")
        parsed_questions = parse_json_object_arg(
            questions,
            option_name="--questions",
        )
        if not parsed_questions:
            raise usage_error("--questions must not be empty")
        result = obj.client.decision.create(
            model=model,
            state=parsed_state,
            questions=parsed_questions,
        )
        renderer.print_rows(
            _rows(result),
            columns=_COLUMNS,
            envelope=result,
        )
        if renderer.table_mode and result.usage is not None:
            renderer.note(
                f"usage: {result.usage.model_dump(exclude_none=True)}",
            )

    return app
