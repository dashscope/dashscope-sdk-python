# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope multimodal`` -- chat create and embeddings create."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.chat.completions import ChatStream
from dashscope.resources.embeddings import Embedding, EmbeddingResponse
from dashscope.types.common import Usage
from dashscope.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    Delta,
    StreamChoice,
)


def _completion(
    content: str,
    *,
    usage: Optional[Dict[str, int]] = None,
) -> ChatCompletion:
    return ChatCompletion(
        model="qwen-vl-max",
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
                finish_reason="stop",
            ),
        ],
        usage=Usage(**usage) if usage else None,
    )


def _chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        choices=[StreamChoice(delta=Delta(role="assistant", content=content))],
    )


class _Inner:
    def __init__(self, chunks: List[ChatCompletionChunk]) -> None:
        self._chunks = chunks
        self.headers: Dict[str, str] = {}
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class _Chat:
    """Stand-in for ``client.multimodal.chat``."""

    def __init__(self, result_fn) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._result_fn = result_fn

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result_fn(kwargs)


class _Embeddings:
    """Stand-in for ``client.multimodal.embeddings``."""

    def __init__(self, response: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _response(count: int = 2, dims: int = 3) -> EmbeddingResponse:
    return EmbeddingResponse(
        embeddings=[
            Embedding(index=index, embedding=[0.1] * dims)
            for index in range(count)
        ],
        usage=Usage(input_tokens=10, total_tokens=10),
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
def multimodal(make_client):
    """Install a fake ``client.multimodal``; returns the two recorders."""

    def install(chat_fn=lambda kwargs: _completion("hello"), embeddings=None):
        chat = _Chat(chat_fn)
        emb = _Embeddings(
            embeddings if embeddings is not None else _response(),
        )
        make_client(multimodal=SimpleNamespace(chat=chat, embeddings=emb))
        return SimpleNamespace(chat=chat, embeddings=emb)

    return install


class TestChatCreate:
    def test_prints_the_answer(self, invoke, multimodal):
        resource = multimodal()
        result = invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "what is this?",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert resource.chat.calls[0]["model"] == "qwen-vl-max"

    def test_falls_back_to_the_configured_default_model(
        self,
        invoke,
        multimodal,
        clean_env,
    ):
        from dashscope.cli import config_store

        config_store.set_value("default_model", "qwen-vl-plus")
        resource = multimodal()
        invoke(["multimodal", "chat", "create", "--prompt", "hi"])
        assert resource.chat.calls[0]["model"] == "qwen-vl-plus"

    def test_no_model_anywhere_is_a_usage_error(self, invoke, multimodal):
        multimodal()
        result = invoke(["multimodal", "chat", "create", "--prompt", "hi"])
        assert result.exit_code == 2
        assert "default_model" in result.stderr

    def test_media_parts_are_passed_through_untouched(
        self,
        invoke,
        multimodal,
    ):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--messages",
                '[{"role": "user", "content": '
                '[{"image": "cat.png"}, {"text": "what is this?"}]}]',
            ],
        )
        assert resource.chat.calls[0]["messages"] == [
            {
                "role": "user",
                "content": [
                    {"image": "cat.png"},
                    {"text": "what is this?"},
                ],
            },
        ]

    def test_prompt_and_system_build_a_conversation(self, invoke, multimodal):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
                "--system",
                "be brief",
            ],
        )
        assert resource.chat.calls[0]["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_sampling_flags_reach_the_sdk(self, invoke, multimodal):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
                "--temperature",
                "0.2",
                "--top-p",
                "0.9",
                "--top-k",
                "5",
                "--max-tokens",
                "64",
                "--stop",
                "END",
                "--seed",
                "7",
                "--enable-thinking",
                "--no-enable-search",
            ],
        )
        kwargs = resource.chat.calls[0]
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 5
        assert kwargs["max_tokens"] == 64
        assert kwargs["stop"] == ["END"]
        assert kwargs["seed"] == 7
        assert kwargs["enable_thinking"] is True
        assert kwargs["enable_search"] is False

    def test_tools_must_be_an_array(self, invoke, multimodal):
        multimodal()
        result = invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
                "--tools",
                '{"type": "function"}',
            ],
        )
        assert result.exit_code == 2
        assert "--tools" in result.stderr

    def test_usage_goes_to_stderr(self, invoke, multimodal):
        multimodal(
            chat_fn=lambda kwargs: _completion(
                "hello",
                usage={"input_tokens": 3, "output_tokens": 4},
            ),
        )
        result = invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
            ],
        )
        assert result.stdout.strip() == "hello"
        assert "input_tokens" in result.stderr

    def test_json_output_is_the_whole_completion(self, invoke, multimodal):
        multimodal()
        result = invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
            ],
            global_args=["-o", "json"],
        )
        assert json.loads(result.stdout)["choices"][0]["message"][
            "content"
        ] == ("hello")

    def test_stream_writes_incremental_text(self, invoke, multimodal):
        def build(kwargs):
            build.inner = _Inner([_chunk(c) for c in ("Hel", "lo")])
            return ChatStream(build.inner)

        build.inner = None
        multimodal(chat_fn=build)
        result = invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
                "--stream",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout == "Hello\n"
        assert build.inner.closed is True

    def test_stream_json_is_one_object_per_line(self, invoke, multimodal):
        multimodal(
            chat_fn=lambda kwargs: ChatStream(
                _Inner([_chunk("Hel"), _chunk("lo")]),
            ),
        )
        result = invoke(
            [
                "multimodal",
                "chat",
                "create",
                "-m",
                "qwen-vl-max",
                "--prompt",
                "hi",
                "--stream",
            ],
            global_args=["-o", "json"],
        )
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        assert [line["choices"][0]["delta"]["content"] for line in lines] == [
            "Hel",
            "lo",
        ]


class TestEmbeddingsCreate:
    def test_text_flag_builds_one_item(self, invoke, multimodal):
        resource = multimodal()
        result = invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                "a cat",
            ],
        )
        assert result.exit_code == 0
        assert resource.embeddings.calls[0]["input"] == [{"text": "a cat"}]

    def test_repeated_flags_build_one_item_each(self, invoke, multimodal):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                "a cat",
                "--text",
                "a dog",
                "--image",
                "cat.png",
            ],
        )
        assert resource.embeddings.calls[0]["input"] == [
            {"text": "a cat"},
            {"text": "a dog"},
            {"image": "cat.png"},
        ]

    def test_text_accepts_an_at_file(self, invoke, multimodal, tmp_path):
        payload = tmp_path / "text.txt"
        payload.write_text("a cat\n", encoding="utf-8")
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                f"@{payload}",
            ],
        )
        assert resource.embeddings.calls[0]["input"] == [{"text": "a cat\n"}]

    def test_input_json_wins_over_the_flags(self, invoke, multimodal):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                "ignored",
                "--input",
                '[{"image": "a.png", "factor": 2}]',
            ],
        )
        assert resource.embeddings.calls[0]["input"] == [
            {"image": "a.png", "factor": 2},
        ]

    def test_input_accepts_a_single_object(self, invoke, multimodal):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--input",
                '{"video": "clip.mp4"}',
            ],
        )
        assert resource.embeddings.calls[0]["input"] == [{"video": "clip.mp4"}]

    def test_input_rejects_a_scalar(self, invoke, multimodal):
        multimodal()
        result = invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--input",
                '"a cat"',
            ],
        )
        assert result.exit_code == 2
        assert "--input" in result.stderr

    def test_input_rejects_a_non_object_entry(self, invoke, multimodal):
        multimodal()
        result = invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--input",
                '["a cat"]',
            ],
        )
        assert result.exit_code == 2
        assert "entry 0" in result.stderr

    def test_nothing_to_embed_is_a_usage_error(self, invoke, multimodal):
        resource = multimodal()
        result = invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
            ],
        )
        assert result.exit_code == 2
        assert "--text" in result.stderr
        assert resource.embeddings.calls == []

    def test_parameters_reach_the_sdk(self, invoke, multimodal):
        resource = multimodal()
        invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                "a cat",
                "--dimension",
                "512",
                "--fps",
                "2.5",
                "--res-level",
                "1",
                "--max-video-frames",
                "80",
                "--enable-fusion",
                "--instruct",
                "retrieve",
                "--parameters",
                '{"output_type": "dense"}',
            ],
        )
        kwargs = resource.embeddings.calls[0]
        assert kwargs["dimension"] == 512
        assert kwargs["fps"] == 2.5
        assert kwargs["res_level"] == 1
        assert kwargs["max_video_frames"] == 80
        assert kwargs["enable_fusion"] is True
        assert kwargs["instruct"] == "retrieve"
        assert kwargs["parameters"] == {"output_type": "dense"}

    def test_table_output_summarises(self, invoke, multimodal):
        multimodal()
        result = invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                "a cat",
                "--text",
                "a dog",
            ],
        )
        assert result.stdout.strip() == "2 embedding(s), 3 dimensions"
        assert "'input_tokens': 10" in result.stderr

    def test_json_output_is_the_whole_response(self, invoke, multimodal):
        multimodal()
        result = invoke(
            [
                "multimodal",
                "embeddings",
                "create",
                "-m",
                "multimodal-embedding-v1",
                "--text",
                "a cat",
            ],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert len(doc["embeddings"]) == 2
        assert doc["usage"]["input_tokens"] == 10
