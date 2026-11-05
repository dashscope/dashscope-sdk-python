# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Output rendering for the dashscope CLI.

One rule drives the whole module: **machine formats never pass through
Rich.** Rich wraps text at the terminal width, which silently corrupts
JSON for any consumer downstream -- a long ``--output json`` payload
would arrive with newlines inserted mid-string. So json/yaml/NDJSON go
out through ``click.echo`` untouched and only the human ``table`` format
gets Rich styling.

That also means a piped ``dashscope ... --output json | jq`` needs no
unwrap workaround, and neither do the tests.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional, Sequence

import click
from rich.console import Console
from rich.table import Table

JSON = "json"
YAML = "yaml"
TABLE = "table"
OUTPUT_FORMATS = (TABLE, JSON, YAML)

# Transport metadata every v2 response carries. It is noise for a human
# and churn for a diff, so it is dropped unless --verbose asked for it.
_TRANSPORT_FIELDS = frozenset({"headers"})


def is_machine_format(fmt: str) -> bool:
    return fmt in (JSON, YAML)


def _strip_transport(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_transport(item)
            for key, item in value.items()
            if key not in _TRANSPORT_FIELDS
        }
    if isinstance(value, list):
        return [_strip_transport(item) for item in value]
    return value


def to_serializable(obj: Any, *, verbose: bool = False) -> Any:
    """Flatten an SDK response into JSON-ready primitives.

    Accepts pydantic models (every v2 response), AgentStudio's typed
    containers, plain containers, and scalars. ``model_dump(mode="json")``
    already converts datetimes and enums, so nothing downstream has to
    know about SDK types.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        value = dump(mode="json", exclude_none=True)
    elif callable(getattr(obj, "to_dict", None)):
        # AgentStudio ships its own lightweight containers instead of
        # pydantic models. ``to_dict`` already recurses into nested
        # containers and its enums are ``StrEnum``, so the result is
        # JSON-ready as it stands.
        value = obj.to_dict()
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        value = obj
    elif isinstance(obj, dict):
        value = {
            key: to_serializable(item, verbose=verbose)
            for key, item in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        value = [to_serializable(item, verbose=verbose) for item in obj]
    else:
        value = str(obj)
    return value if verbose else _strip_transport(value)


class Renderer:
    """Formats one command's result according to ``--output``."""

    def __init__(
        self,
        fmt: str = TABLE,
        *,
        no_color: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        if fmt not in OUTPUT_FORMATS:
            raise ValueError(
                f"invalid output format {fmt!r}, "
                f"expected one of {OUTPUT_FORMATS}",
            )
        self.fmt = fmt
        self.verbose = verbose
        if no_color is None:
            no_color = bool(os.environ.get("NO_COLOR"))
        self.no_color = no_color
        self._out = Console(no_color=no_color)
        self._err = Console(stderr=True, no_color=no_color)

    # -- raw channels ----------------------------------------------------

    def echo(self, text: str = "") -> None:
        """Plain stdout: no markup, no wrapping. Use for payload text."""
        click.echo(text)

    def write(self, text: str) -> None:
        """Unbuffered stdout write for incremental streaming."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def note(self, text: str) -> None:
        """Progress/diagnostic text on stderr, so stdout stays pipeable."""
        self._err.print(f"[dim]{text}[/dim]", highlight=False)

    def success(self, text: str) -> None:
        if is_machine_format(self.fmt):
            return
        self._err.print(f"[green]\u2713[/green] {text}")

    def warn(self, text: str) -> None:
        self._err.print(f"[yellow]{text}[/yellow]")

    @property
    def table_mode(self) -> bool:
        """True when human-facing Rich rendering is appropriate."""
        return self.fmt == TABLE

    @property
    def err_console(self) -> Console:
        """The stderr console, for callers that render progress themselves.

        Already carries the resolved ``no_color`` decision, so a spinner
        or live status cannot leak colour into a redirected log.
        """
        return self._err

    # -- structured channels ---------------------------------------------

    def print_json(self, obj: Any) -> None:
        payload = to_serializable(obj, verbose=self.verbose)
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))

    def print_ndjson(self, obj: Any) -> None:
        """One compact, valid JSON object per line (jq-consumable)."""
        payload = to_serializable(obj, verbose=self.verbose)
        click.echo(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def print_yaml(self, obj: Any) -> None:
        # Imported lazily: PyYAML is a declared dependency but --output
        # json is the common path and should not pay for it.
        import yaml  # pylint: disable=import-outside-toplevel

        payload = to_serializable(obj, verbose=self.verbose)
        click.echo(
            yaml.safe_dump(
                payload,
                allow_unicode=True,
                sort_keys=False,
            ).rstrip(),
        )

    def print_object(
        self,
        obj: Any,
        *,
        columns: Optional[Sequence[str]] = None,
    ):
        """Render a single resource according to ``--output``."""
        if self.fmt == JSON:
            self.print_json(obj)
        elif self.fmt == YAML:
            self.print_yaml(obj)
        else:
            self._print_fields(obj, columns=columns)

    def print_rows(
        self,
        rows: Sequence[Any],
        *,
        columns: Sequence[str],
        envelope: Any = None,
    ) -> None:
        """Render a list endpoint.

        Machine formats get the whole envelope (pagination metadata
        included) so a caller can page without a second request; the
        human format gets a table of just the rows.
        """
        if self.fmt == JSON:
            self.print_json(envelope if envelope is not None else rows)
            return
        if self.fmt == YAML:
            self.print_yaml(envelope if envelope is not None else rows)
            return
        self._print_table(rows, columns=columns)

    def print_stream_chunk(self, obj: Any) -> None:
        """Emit one streaming frame in the active machine format.

        YAML streams as NDJSON too: a multi-document YAML stream is legal
        but no consumer expects one, and the frames are independent
        objects either way.
        """
        self.print_ndjson(obj)

    # -- rich internals ---------------------------------------------------

    def _print_fields(
        self,
        obj: Any,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        data = to_serializable(obj, verbose=self.verbose)
        if not isinstance(data, dict):
            click.echo(data)
            return
        keys = list(columns) if columns else list(data)
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("field", style="cyan", no_wrap=True)
        table.add_column("value", overflow="fold")
        for key in keys:
            if key not in data:
                continue
            table.add_row(key, _cell(data[key]))
        self._out.print(table)

    def _print_table(self, rows: Sequence[Any], *, columns: Sequence[str]):
        payload = [to_serializable(row, verbose=self.verbose) for row in rows]
        table = Table(box=None, pad_edge=False)
        for column in columns:
            table.add_column(column, overflow="fold")
        for row in payload:
            if isinstance(row, dict):
                table.add_row(*[_cell(row.get(column)) for column in columns])
            else:
                table.add_row(_cell(row))
        self._out.print(table)


def render_completion(renderer: Renderer, result: Any) -> None:
    """One chat completion: payload on stdout, diagnostics on stderr.

    Keeping reasoning and usage off stdout is what makes
    ``dashscope chat create ... > answer.txt`` usable.
    """
    if not renderer.table_mode:
        renderer.print_object(result)
        return
    for choice in result.choices:
        message = choice.message
        if message.reasoning_content:
            renderer.note(message.reasoning_content)
        renderer.echo(message.content or "")
    if result.usage is not None:
        renderer.note(f"usage: {result.usage.model_dump(exclude_none=True)}")


def render_text_stream(renderer: Renderer, stream: Any) -> None:
    """A chat-shaped stream (``text_stream`` plus whole-chunk iteration).

    Deliberately has no ``except``: a mid-stream failure must propagate to
    the error handler and exit non-zero. The 1.x tree caught these,
    printed a message, and returned -- which exits 0.
    """
    if renderer.table_mode:
        _write_stream_text(renderer, stream)
        renderer.echo()
        return
    for chunk in stream:
        renderer.print_stream_chunk(chunk)


def _write_stream_text(renderer: Renderer, stream: Any) -> None:
    """Append the answer to the terminal, whatever the frame shape.

    A ``cumulative`` stream (``incremental_output=False``, or a protocol
    the SDK merges client-side) yields the whole text so far on every
    frame, so only the tail that has not been written yet may go out --
    appending snapshots verbatim repeats the answer. Read with a default
    because not every chat-shaped stream carries the flag.
    """
    if not getattr(stream, "cumulative", False):
        for text in stream.text_stream:
            renderer.write(text)
        return
    written = 0
    for text in stream.text_stream:
        renderer.write(text[written:])
        written = max(written, len(text))


def render_stream_event(renderer: Renderer, event: Any) -> None:
    """One server-sent event, in whichever format is active.

    No ``except`` here or in the callers: ``stream_events`` raises on an
    error frame (``raise_on_error`` is on by default), and that has to
    reach the error handler. The 1.x tree caught it, printed, and returned
    -- which exits 0.
    """
    if not renderer.table_mode:
        renderer.print_stream_chunk(event)
        return
    output = event.output
    if isinstance(output, str):
        renderer.echo(output)
    elif output is None:
        renderer.echo("")
    else:
        renderer.echo(
            json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        )


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def resolve_no_color(flag: bool) -> bool:
    """``--no-color`` or a non-empty ``NO_COLOR`` environment variable."""
    return bool(flag) or bool(os.environ.get("NO_COLOR"))
