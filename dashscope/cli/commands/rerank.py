# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope rerank`` -- document reranking.

One synchronous call: give it a query and a list of documents, get back
a relevance-sorted list. The CLI accepts documents as a JSON array or
as repeatable ``--document`` flags, whichever is easier in a pipeline.
"""

from __future__ import annotations

from typing import List, Optional

import typer

from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_text_arg,
)

_HELP = r"""Rerank documents by relevance to a query.

`rerank create` takes a query and a list of documents, and returns them
sorted by relevance score. Documents can be passed as a JSON array via
`--documents`, or one at a time via repeatable `--document` flags.
"""

_RESULT_COLUMNS = (
    "index",
    "relevance_score",
)


def _parse_documents(
    raw: Optional[str],
    repeatable: Optional[List[str]],
) -> List[str]:
    """Merge --documents (JSON array) and --document (repeatable) into one list."""
    docs: List[str] = []
    if raw is not None:
        text = read_text_arg(raw, option_name="--documents")
        if text is not None:
            value = parse_json_arg(raw, option_name="--documents")
            if not isinstance(value, list):
                raise usage_error(
                    "--documents: expected a JSON array of strings"
                )
            docs.extend(str(item) for item in value)
    if repeatable:
        docs.extend(repeatable)
    return docs


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("create")
    def create(
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Rerank model name.",
        ),
        query: str = typer.Option(
            ...,
            "-q",
            "--query",
            help="The query to rank documents against. Accepts @file and stdin (-).",
        ),
        documents: Optional[str] = typer.Option(
            None,
            "--documents",
            help="Documents as a JSON array of strings. Accepts @file.",
        ),
        document: Optional[List[str]] = typer.Option(
            None,
            "--document",
            "-d",
            help="One document. Repeatable.",
        ),
        top_n: Optional[int] = typer.Option(
            None,
            "--top-n",
            help="Return only the top N results.",
        ),
        return_documents: Optional[bool] = typer.Option(
            None,
            "--return-documents/--no-return-documents",
            help="Include the document text in each result.",
        ),
        instruct: Optional[str] = typer.Option(
            None,
            "--instruct",
            help="Instruction prefix for the rerank model.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Rerank documents by relevance."""
        obj = ctx.obj
        renderer = obj.renderer
        query_text = read_text_arg(query, option_name="--query")
        if not query_text:
            raise usage_error("--query is required")
        docs = _parse_documents(documents, document)
        if not docs:
            raise usage_error(
                "no documents given: pass --documents (JSON array) "
                "or repeatable --document",
            )
        result = obj.client.rerank.create(
            model=model,
            query=query_text,
            documents=docs,
            top_n=top_n,
            return_documents=return_documents,
            instruct=instruct,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if renderer.table_mode:
            renderer.print_rows(result.results, columns=_RESULT_COLUMNS)
            if result.usage:
                renderer.note(
                    f"usage: total_tokens={result.usage.total_tokens}",
                )
        else:
            renderer.print_object(result)

    return app
