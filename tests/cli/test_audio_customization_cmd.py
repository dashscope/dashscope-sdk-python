# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope audio`` management plane: vocabularies, phrases, voices."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.audio.customization import (
    DesignedVoice,
    DesignedVoiceList,
    Vocabulary,
    VocabularyList,
    Voice,
    VoiceList,
)
from dashscope.resources.audio.phrases import Phrase, PhraseList
from dashscope.types.common import JobStatus

_ENTRIES = [{"text": "心肌梗死", "weight": 4, "lang": "zh"}]


def _vocabulary(**overrides: Any) -> Vocabulary:
    data: Dict[str, Any] = {
        "vocabulary_id": "vocab-med-1",
        "prefix": "med",
        "target_model": "paraformer-v2",
        "status": "OK",
        "request_id": "req-1",
    }
    data.update(overrides)
    return Vocabulary(**data)


def _voice(**overrides: Any) -> Voice:
    data: Dict[str, Any] = {
        "voice_id": "cosyvoice-v2-mine-1",
        "status": "OK",
        "target_model": "cosyvoice-v2",
        "request_id": "req-1",
    }
    data.update(overrides)
    return Voice(**data)


def _designed_voice(**overrides: Any) -> DesignedVoice:
    data: Dict[str, Any] = {
        "voice_id": "cosyvoice-v3-plus-design-1",
        "status": "DEPLOYING",
        "target_model": "cosyvoice-v3-plus",
        "voice_prompt": "a calm narrator",
        "preview_text": "hello",
        "request_id": "req-1",
    }
    data.update(overrides)
    return DesignedVoice(**data)


def _phrase(**overrides: Any) -> Phrase:
    data: Dict[str, Any] = {
        "phrase_id": "phrase-1",
        "status": "OK",
        "request_id": "req-1",
    }
    data.update(overrides)
    return Phrase(**data)


class _Recorder:
    """One registry stand-in: every method records and returns its value."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def _answer(self, method: str, default: Any) -> Any:
        return self._results.get(method, default)

    def create(self, **kwargs):
        self.calls.append({"method": "create", "args": (), "kwargs": kwargs})
        return self._answer("create", self._default_create())

    def list(self, **kwargs):  # pylint: disable=redefined-builtin
        self.calls.append({"method": "list", "args": (), "kwargs": kwargs})
        return self._answer("list", self._default_list())

    def get(self, resource_id, **kwargs):
        self.calls.append(
            {"method": "get", "args": (resource_id,), "kwargs": kwargs},
        )
        return self._answer("get", self._default_create())

    def update(self, resource_id, **kwargs):
        self.calls.append(
            {"method": "update", "args": (resource_id,), "kwargs": kwargs},
        )
        return self._answer("update", JobStatus(status="OK"))

    def delete(self, resource_id, **kwargs):
        self.calls.append(
            {"method": "delete", "args": (resource_id,), "kwargs": kwargs},
        )
        return self._answer("delete", JobStatus(status="DELETED"))

    def _default_create(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def _default_list(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


class _Vocabularies(_Recorder):
    def _default_create(self) -> Any:
        return _vocabulary()

    def _default_list(self) -> Any:
        return VocabularyList(
            vocabularies=[
                {
                    "vocabulary_id": "vocab-med-1",
                    "prefix": "med",
                    "target_model": "paraformer-v2",
                    "status": "OK",
                },
            ],
        )


class _Voices(_Recorder):
    def _default_create(self) -> Any:
        return _voice()

    def _default_list(self) -> Any:
        return VoiceList(
            voices=[
                {
                    "voice_id": "cosyvoice-v2-mine-1",
                    "status": "OK",
                    "target_model": "cosyvoice-v2",
                },
            ],
        )


class _VoiceDesign(_Recorder):
    def _default_create(self) -> Any:
        return _designed_voice()

    def _default_list(self) -> Any:
        return DesignedVoiceList(
            voices=[
                {
                    "voice_id": "cosyvoice-v3-plus-design-1",
                    "status": "DEPLOYING",
                    "target_model": "cosyvoice-v3-plus",
                },
            ],
        )

    def update(self, *args, **kwargs):  # pragma: no cover - not on the wire
        raise AssertionError("voice-design has no update")


class _Phrases(_Recorder):
    def _default_create(self) -> Any:
        return _phrase()

    def _default_list(self) -> Any:
        return PhraseList(
            page_no=1,
            page_size=10,
            total_count=2,
            phrases=[_phrase(), _phrase(phrase_id="phrase-2")],
        )


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def registries(make_client):
    """Install all four registries and hand them back by name."""

    def install(**overrides: Any) -> SimpleNamespace:
        vocabularies = _Vocabularies(**overrides.pop("vocabularies", {}))
        phrases = _Phrases(**overrides.pop("phrases", {}))
        voices = _Voices(**overrides.pop("voices", {}))
        voice_design = _VoiceDesign(**overrides.pop("voice_design", {}))
        audio = SimpleNamespace(
            vocabularies=vocabularies,
            phrases=phrases,
            voices=voices,
            voice_design=voice_design,
        )
        make_client(audio=audio)
        return audio

    return install


class TestVocabularies:
    def test_create_sends_the_entries(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "vocabularies",
                "create",
                "--target-model",
                "paraformer-v2",
                "--prefix",
                "med",
                "--vocabulary",
                json.dumps(_ENTRIES),
            ],
        )
        assert result.exit_code == 0
        assert "vocab-med-1" in result.stdout
        kwargs = audio.vocabularies.calls[0]["kwargs"]
        assert kwargs["target_model"] == "paraformer-v2"
        assert kwargs["prefix"] == "med"
        assert kwargs["vocabulary"] == _ENTRIES
        assert kwargs["model"] is None

    def test_create_reads_the_entries_from_a_file(
        self,
        invoke,
        registries,
        tmp_path,
    ):
        payload = tmp_path / "words.json"
        payload.write_text(json.dumps(_ENTRIES), encoding="utf-8")
        audio = registries()
        invoke(
            [
                "audio",
                "vocabularies",
                "create",
                "--target-model",
                "paraformer-v2",
                "--prefix",
                "med",
                "--vocabulary",
                f"@{payload}",
            ],
        )
        assert audio.vocabularies.calls[0]["kwargs"]["vocabulary"] == _ENTRIES

    def test_create_rejects_a_json_object(self, invoke, registries):
        """The service wants an array of entries, not one entry."""
        audio = registries()
        result = invoke(
            [
                "audio",
                "vocabularies",
                "create",
                "--target-model",
                "paraformer-v2",
                "--prefix",
                "med",
                "--vocabulary",
                '{"text": "x"}',
            ],
        )
        assert result.exit_code == 2
        assert "expected a JSON array" in result.stderr
        assert audio.vocabularies.calls == []

    def test_list_pages_from_zero(self, invoke, registries):
        audio = registries()
        result = invoke(
            ["audio", "vocabularies", "list", "--prefix", "med"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        kwargs = audio.vocabularies.calls[0]["kwargs"]
        assert kwargs == {"prefix": "med", "page_index": 0, "page_size": 10}
        assert json.loads(result.stdout)["vocabularies"][0]["prefix"] == "med"

    def test_get_shows_one_table(self, invoke, registries):
        audio = registries()
        result = invoke(["audio", "vocabularies", "get", "vocab-med-1"])
        assert result.exit_code == 0
        assert "vocab-med-1" in result.stdout
        assert audio.vocabularies.calls[0]["args"] == ("vocab-med-1",)

    def test_update_replaces_the_contents(self, invoke, registries):
        audio = registries()
        invoke(
            [
                "audio",
                "vocabularies",
                "update",
                "vocab-med-1",
                "--vocabulary",
                json.dumps(_ENTRIES),
            ],
        )
        call = audio.vocabularies.calls[0]
        assert call["args"] == ("vocab-med-1",)
        assert call["kwargs"]["vocabulary"] == _ENTRIES

    def test_delete_needs_a_confirmation(self, invoke, registries):
        audio = registries()
        result = invoke(["audio", "vocabularies", "delete", "vocab-med-1"])
        assert result.exit_code != 0
        assert audio.vocabularies.calls == []

    def test_delete_with_yes_goes_through(self, invoke, registries):
        audio = registries()
        result = invoke(
            ["audio", "vocabularies", "delete", "vocab-med-1", "--yes"],
        )
        assert result.exit_code == 0
        assert "DELETED" in result.stdout
        assert audio.vocabularies.calls[0]["method"] == "delete"


class TestPhrases:
    def test_create_compiles_a_phrase_set(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "phrases",
                "create",
                "--phrases",
                '{"class_a": ["alpha", "beta"]}',
            ],
        )
        assert result.exit_code == 0
        assert "phrase-1" in result.stdout
        kwargs = audio.phrases.calls[0]["kwargs"]
        assert kwargs["phrases"] == {"class_a": ["alpha", "beta"]}
        assert kwargs["training_type"] == "compile_asr_phrase"

    def test_create_rejects_an_empty_set(self, invoke, registries):
        audio = registries()
        result = invoke(["audio", "phrases", "create", "--phrases", "{}"])
        assert result.exit_code == 2
        assert "must not be empty" in result.stderr
        assert audio.phrases.calls == []

    def test_list_pages_from_one(self, invoke, registries):
        """This endpoint is fine-tune-shaped, unlike its three siblings."""
        audio = registries()
        result = invoke(["audio", "phrases", "list", "--page", "2"])
        assert result.exit_code == 0
        assert audio.phrases.calls[0]["kwargs"] == {
            "page_no": 2,
            "page_size": 10,
        }
        assert "page 1/1, total 2" in result.stderr

    def test_update_recompiles(self, invoke, registries):
        audio = registries()
        invoke(
            [
                "audio",
                "phrases",
                "update",
                "phrase-1",
                "--phrases",
                '{"class_a": ["gamma"]}',
                "--training-type",
                "compile_asr_phrase_v2",
            ],
        )
        call = audio.phrases.calls[0]
        assert call["args"] == ("phrase-1",)
        assert call["kwargs"]["training_type"] == "compile_asr_phrase_v2"

    def test_delete_with_yes_goes_through(self, invoke, registries):
        audio = registries()
        result = invoke(["audio", "phrases", "delete", "phrase-1", "-y"])
        assert result.exit_code == 0
        assert audio.phrases.calls[0]["method"] == "delete"


class TestVoices:
    def test_create_clones_from_a_sample(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "voices",
                "create",
                "--target-model",
                "cosyvoice-v2",
                "--prefix",
                "mine",
                "--url",
                "https://example.com/sample.wav",
                "--language-hint",
                "zh",
                "--max-prompt-audio-length",
                "8.5",
                "--extra-input",
                '{"a": 1}',
            ],
        )
        assert result.exit_code == 0
        assert "cosyvoice-v2-mine-1" in result.stdout
        kwargs = audio.voices.calls[0]["kwargs"]
        assert kwargs["target_model"] == "cosyvoice-v2"
        assert kwargs["prefix"] == "mine"
        assert kwargs["url"] == "https://example.com/sample.wav"
        assert kwargs["language_hints"] == ["zh"]
        assert kwargs["max_prompt_audio_length"] == 8.5
        assert kwargs["extra_input"] == {"a": 1}

    def test_unset_options_stay_unset(self, invoke, registries):
        audio = registries()
        invoke(
            [
                "audio",
                "voices",
                "create",
                "--target-model",
                "cosyvoice-v2",
                "--prefix",
                "mine",
                "--url",
                "https://example.com/sample.wav",
            ],
        )
        kwargs = audio.voices.calls[0]["kwargs"]
        assert kwargs["language_hints"] is None
        assert kwargs["max_prompt_audio_length"] is None
        assert kwargs["extra_input"] is None

    def test_list_shares_the_registry_with_design(self, invoke, registries):
        audio = registries()
        result = invoke(
            ["audio", "voices", "list"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        assert audio.voices.calls[0]["kwargs"] == {
            "prefix": None,
            "page_index": 0,
            "page_size": 10,
        }
        assert json.loads(result.stdout)["voices"][0]["voice_id"] == (
            "cosyvoice-v2-mine-1"
        )

    def test_update_re_clones(self, invoke, registries):
        audio = registries()
        invoke(
            [
                "audio",
                "voices",
                "update",
                "cosyvoice-v2-mine-1",
                "--url",
                "https://example.com/better.wav",
            ],
        )
        call = audio.voices.calls[0]
        assert call["args"] == ("cosyvoice-v2-mine-1",)
        assert call["kwargs"] == {"url": "https://example.com/better.wav"}

    def test_delete_with_yes_goes_through(self, invoke, registries):
        audio = registries()
        result = invoke(
            ["audio", "voices", "delete", "cosyvoice-v2-mine-1", "--yes"],
        )
        assert result.exit_code == 0
        assert audio.voices.calls[0]["method"] == "delete"


class TestVoiceDesign:
    def test_create_sends_both_texts(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "voice-design",
                "create",
                "--target-model",
                "cosyvoice-v3-plus",
                "--prefix",
                "design",
                "--voice-prompt",
                "a calm narrator",
                "--preview-text",
                "hello there",
                "--language-hint",
                "en",
            ],
        )
        assert result.exit_code == 0
        assert "DEPLOYING" in result.stdout
        kwargs = audio.voice_design.calls[0]["kwargs"]
        assert kwargs["voice_prompt"] == "a calm narrator"
        assert kwargs["preview_text"] == "hello there"
        assert kwargs["language_hints"] == ["en"]

    def test_create_reads_the_prompt_from_stdin(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "voice-design",
                "create",
                "--target-model",
                "cosyvoice-v3-plus",
                "--prefix",
                "design",
                "--voice-prompt",
                "-",
                "--preview-text",
                "hello",
            ],
            input="a warm voice\n",
        )
        assert result.exit_code == 0
        assert audio.voice_design.calls[0]["kwargs"]["voice_prompt"] == (
            "a warm voice\n"
        )

    def test_an_empty_prompt_is_a_usage_error(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "voice-design",
                "create",
                "--target-model",
                "cosyvoice-v3-plus",
                "--prefix",
                "design",
                "--voice-prompt",
                "   ",
                "--preview-text",
                "hello",
            ],
        )
        assert result.exit_code == 2
        assert "--voice-prompt must not be empty" in result.stderr
        assert audio.voice_design.calls == []

    def test_get_polls_until_it_is_usable(self, invoke, registries):
        """The docstring tells the caller to poll; `get` is how."""
        audio = registries(
            voice_design={"get": _designed_voice(status="OK")},
        )
        result = invoke(
            ["audio", "voice-design", "get", "cosyvoice-v3-plus-design-1"],
        )
        assert result.exit_code == 0
        assert "OK" in result.stdout
        assert audio.voice_design.calls[0]["method"] == "get"

    def test_delete_with_yes_goes_through(self, invoke, registries):
        audio = registries()
        result = invoke(
            [
                "audio",
                "voice-design",
                "delete",
                "cosyvoice-v3-plus-design-1",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert audio.voice_design.calls[0]["method"] == "delete"


class TestGroupShape:
    def test_the_four_registries_are_subgroups_of_audio(self, app):
        import typer

        audio = typer.main.get_command(app).commands["audio"]
        for name in ("vocabularies", "phrases", "voices", "voice-design"):
            assert name in audio.commands

    def test_voice_design_has_no_update(self, app):
        """The service has none, so the CLI must not invent one."""
        import typer

        audio = typer.main.get_command(app).commands["audio"]
        assert "update" not in audio.commands["voice-design"].commands

    def test_bare_subgroup_shows_help(self, app, runner):
        assert "Usage" in runner.invoke(app, ["audio", "vocabularies"]).stdout
