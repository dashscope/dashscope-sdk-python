# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Output rendering: machine formats must stay machine-readable."""

import json

import pytest

from dashscope.cli2.output import (
    JSON,
    OUTPUT_FORMATS,
    TABLE,
    YAML,
    Renderer,
    is_machine_format,
    resolve_no_color,
    to_serializable,
)


class _Model:
    """A stand-in for a pydantic response: only ``model_dump`` matters."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, **kwargs):
        assert kwargs.get("mode") == "json"
        assert kwargs.get("exclude_none") is True
        return self._payload


class TestToSerializable:
    def test_model_dump_is_used(self):
        assert to_serializable(_Model({"a": 1})) == {"a": 1}

    def test_headers_are_transport_metadata(self):
        model = _Model({"a": 1, "headers": {"x-request-id": "r"}})
        assert to_serializable(model) == {"a": 1}
        assert to_serializable(model, verbose=True) == {
            "a": 1,
            "headers": {"x-request-id": "r"},
        }

    def test_headers_are_stripped_recursively(self):
        payload = {
            "choices": [
                {
                    "message": {"content": "x"},
                    "headers": {"h": "1"},
                },
            ],
        }
        assert to_serializable(_Model(payload)) == {
            "choices": [{"message": {"content": "x"}}],
        }

    def test_scalars_and_containers(self):
        assert to_serializable("s") == "s"
        assert to_serializable(3) == 3
        assert to_serializable(None) is None
        assert to_serializable({"a": [1, {"headers": 2}]}) == {"a": [1, {}]}

    def test_unknown_object_falls_back_to_str(self):
        class Opaque:
            def __str__(self):
                return "opaque"

        assert to_serializable(Opaque()) == "opaque"


class TestMachineFormats:
    def test_formats(self):
        assert OUTPUT_FORMATS == (TABLE, JSON, YAML)
        assert is_machine_format(JSON) and is_machine_format(YAML)
        assert not is_machine_format(TABLE)

    def test_invalid_format_rejected(self):
        with pytest.raises(ValueError):
            Renderer("csv")

    def test_long_json_is_not_wrapped(self, capsys, monkeypatch):
        """The whole point of bypassing Rich.

        Rich wraps at the terminal width, which would insert newlines
        inside a JSON string and hand a downstream consumer invalid JSON.
        """
        monkeypatch.setenv("COLUMNS", "40")
        long_text = "word " * 60
        Renderer(JSON, no_color=True).print_json({"text": long_text})
        out = capsys.readouterr().out
        assert json.loads(out)["text"] == long_text
        # indent=2 yields exactly three physical lines: the value itself
        # was never re-flowed to the terminal width.
        assert len(out.strip().splitlines()) == 3

    def test_json_is_utf8_not_escaped(self, capsys):
        Renderer(JSON, no_color=True).print_json({"text": "你好"})
        assert "你好" in capsys.readouterr().out

    def test_ndjson_is_one_valid_object_per_line(self, capsys):
        renderer = Renderer(JSON, no_color=True)
        for index in range(3):
            renderer.print_ndjson({"i": index, "text": "a" * 100})
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["i"] for line in lines] == [0, 1, 2]

    def test_yaml(self, capsys):
        import yaml

        Renderer(YAML, no_color=True).print_yaml({"a": 1, "b": [1, 2]})
        out = capsys.readouterr().out
        assert yaml.safe_load(out) == {"a": 1, "b": [1, 2]}
        assert not out.endswith("\n\n")

    def test_yaml_preserves_insertion_order(self, capsys):
        import yaml

        Renderer(YAML, no_color=True).print_yaml({"z": 1, "a": 2})
        out = capsys.readouterr().out
        assert out.index("z:") < out.index("a:")
        assert list(yaml.safe_load(out)) == ["z", "a"]

    def test_stream_chunk_is_ndjson_in_both_machine_formats(self, capsys):
        for fmt in (JSON, YAML):
            Renderer(fmt, no_color=True).print_stream_chunk({"delta": "x"})
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line) == {"delta": "x"}


class TestPrintObject:
    def test_json(self, capsys):
        Renderer(JSON, no_color=True).print_object(_Model({"job_id": "j1"}))
        assert json.loads(capsys.readouterr().out) == {"job_id": "j1"}

    def test_yaml(self, capsys):
        Renderer(YAML, no_color=True).print_object(_Model({"job_id": "j1"}))
        assert "job_id: j1" in capsys.readouterr().out

    def test_table_lists_requested_columns_in_order(self, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        Renderer(TABLE, no_color=True).print_object(
            _Model({"b": 2, "a": 1, "c": 3}),
            columns=["a", "c"],
        )
        out = capsys.readouterr().out
        assert "a" in out and "c" in out
        assert "b" not in out

    def test_table_without_columns_shows_everything(self, capsys):
        Renderer(TABLE, no_color=True).print_object(_Model({"a": 1}))
        assert "a" in capsys.readouterr().out


class TestPrintRows:
    def test_machine_format_gets_the_envelope(self, capsys):
        rows = [_Model({"job_id": "j1"})]
        envelope = _Model({"fine_tunes": [{"job_id": "j1"}], "total_count": 7})
        Renderer(JSON, no_color=True).print_rows(
            rows,
            columns=["job_id"],
            envelope=envelope,
        )
        assert json.loads(capsys.readouterr().out)["total_count"] == 7

    def test_machine_format_without_envelope_gets_the_rows(self, capsys):
        Renderer(JSON, no_color=True).print_rows(
            [_Model({"job_id": "j1"})],
            columns=["job_id"],
        )
        assert json.loads(capsys.readouterr().out) == [{"job_id": "j1"}]

    def test_table_format_shows_only_rows(self, capsys):
        rows = [{"job_id": "j1", "status": "PENDING"}]
        Renderer(TABLE, no_color=True).print_rows(
            rows,
            columns=["job_id", "status"],
            envelope={"total_count": 7},
        )
        out = capsys.readouterr().out
        assert "j1" in out and "PENDING" in out
        assert "total_count" not in out

    def test_missing_cell_renders_as_a_dash(self, capsys):
        Renderer(TABLE, no_color=True).print_rows(
            [{"a": None}],
            columns=["a", "b"],
        )
        out = capsys.readouterr().out
        assert out.count("-") == 2

    def test_nested_cell_is_compact_json(self, capsys):
        Renderer(TABLE, no_color=True).print_rows(
            [{"usage": {"total_tokens": 5}}],
            columns=["usage"],
        )
        assert '{"total_tokens":5}' in capsys.readouterr().out


class TestSideChannels:
    def test_echo_goes_to_stdout_unstyled(self, capsys):
        Renderer(TABLE, no_color=True).echo("[red]not markup[/red]")
        result = capsys.readouterr()
        assert result.out == "[red]not markup[/red]\n"
        assert result.err == ""

    def test_write_is_unbuffered_and_newline_free(self, capsys):
        renderer = Renderer(TABLE, no_color=True)
        renderer.write("a")
        renderer.write("b")
        assert capsys.readouterr().out == "ab"

    def test_note_goes_to_stderr(self, capsys):
        Renderer(TABLE, no_color=True).note("progress")
        result = capsys.readouterr()
        assert result.out == ""
        assert "progress" in result.err

    def test_success_is_silent_in_machine_formats(self, capsys):
        for fmt in (JSON, YAML):
            Renderer(fmt, no_color=True).success("done")
        assert capsys.readouterr().err == ""

    def test_success_shows_in_table_format(self, capsys):
        Renderer(TABLE, no_color=True).success("done")
        result = capsys.readouterr()
        assert "done" in result.err
        assert result.out == ""


class TestNoColor:
    def test_flag(self):
        assert resolve_no_color(True) is True

    def test_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert resolve_no_color(False) is True

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert resolve_no_color(False) is False

    def test_renderer_honours_env_by_default(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert Renderer(TABLE).no_color is True
