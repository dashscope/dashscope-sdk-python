# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatible option spellings on commands shared by 1.x and v2."""

import json

import pytest

from dashscope.cli.compat_1x import normalize_argv
from dashscope.cli.errors import CliError


@pytest.mark.parametrize(
    ("argv", "prefix"),
    [
        (["auth", "whoami"], ["auth", "whoami"]),
        (["auth", "login"], ["auth", "login"]),
        (["auth", "logout"], ["auth", "logout"]),
        (["ft", "create"], ["ft", "create"]),
        (["fine-tunes", "call"], ["ft", "create"]),
        (["ft", "get"], ["ft", "get"]),
        (["ft", "list"], ["ft", "list"]),
        (["ft", "stream"], ["ft", "stream"]),
        (["ft", "cancel"], ["ft", "cancel"]),
        (["ft", "delete"], ["ft", "delete"]),
        (["files", "upload"], ["files", "upload"]),
        (["files", "get"], ["files", "get"]),
        (["files", "list"], ["files", "list"]),
        (["files", "delete"], ["files", "delete"]),
        (["deployments", "create"], ["deployments", "create"]),
        (["deployments", "call"], ["deployments", "create"]),
        (["deployments", "get"], ["deployments", "get"]),
        (["deployments", "list"], ["deployments", "list"]),
        (["deployments", "scale"], ["deployments", "scale"]),
        (["deployments", "delete"], ["deployments", "delete"]),
        (["oss", "upload"], ["oss", "upload"]),
        (["rerank", "create"], ["rerank", "create"]),
        (["embeddings", "create"], ["embeddings", "create"]),
        (["tokenization", "create"], ["tokenization", "count"]),
        (["models", "list"], ["models", "list"]),
        (["models", "get"], ["models", "get"]),
        (["rl", "register_functions"], ["rl", "functions", "register"]),
        (["agentic-rl", "test_functions"], ["rl", "functions", "test"]),
        (["rl", "upload_data"], ["rl", "functions", "upload-data"]),
        (["rl", "run"], ["rl", "jobs", "create"]),
        (["rl", "get"], ["rl", "jobs", "get"]),
        (["rl", "logs"], ["rl", "jobs", "logs"]),
        (["rl", "list"], ["rl", "jobs", "list"]),
        (["rl", "cancel"], ["rl", "jobs", "cancel"]),
    ],
)
def test_shared_commands_map_to_v2(argv, prefix):
    assert normalize_argv(argv)[: len(prefix)] == prefix


def test_root_equals_aliases_and_trailing_globals():
    assert normalize_argv(["--api_key=sk-1", "ft", "list"])[:2] == [
        "--api-key=sk-1",
        "ft",
    ]
    assert normalize_argv(["ft", "list", "--base_url=https://example.com"])[
        :2
    ] == ["--base-url", "https://example.com"]


def test_shared_list_options_keep_legacy_spellings():
    for group in ("ft", "files", "deployments", "models"):
        result = normalize_argv(
            [group, "list", "--start_page", "2", "--page_size", "5"],
        )
        assert result[result.index("--page") + 1] == "2"
        assert result[result.index("--size") + 1] == "5"


def test_shared_legacy_defaults_are_preserved():
    assert "--wait" in normalize_argv(["deployments", "call", "-m", "model"])
    assert "--yes" in normalize_argv(["deployments", "delete", "name"])
    assert "--yes" in normalize_argv(["files", "delete", "file-1"])
    assert "--yes" in normalize_argv(["rl", "cancel", "job"])
    assert "--wait" in normalize_argv(["ft", "create"])
    assert "--wait" in normalize_argv(["fine-tunes", "create"])


def test_ft_training_file_options_are_normalised():
    result = normalize_argv(
        [
            "ft",
            "call",
            "--training_file_ids",
            "train-1",
            "--validation_file_ids",
            "valid-1",
            "--n_epochs",
            "2",
            "--learning_rate",
            "0.0001",
        ],
    )
    for option in (
        "--training-file-ids",
        "--validation-file-ids",
        "--n-epochs",
        "--learning-rate",
        "--wait",
    ):
        assert option in result


def test_tokenization_legacy_create_and_enable_search():
    result = normalize_argv(
        [
            "tokenization",
            "create",
            "-m",
            "qwen-plus",
            "-p",
            "hello",
            "--enable-search",
        ],
    )
    assert result[:3] == ["tokenization", "count", "-m"]
    parameters = json.loads(result[result.index("--parameters") + 1])
    assert parameters == {"enable_search": True}


def test_rl_legacy_names_and_config_overrides():
    result = normalize_argv(
        [
            "agentic-rl",
            "run",
            "--config",
            "rl.toml",
            "--teacher-model",
            "qwen-plus",
            "-o",
            "json",
            "-v",
        ],
    )
    assert result[:7] == [
        "--output",
        "json",
        "--verbose",
        "rl",
        "jobs",
        "create",
        "--config",
    ]
    assert result[result.index("--set") + 1] == "teacher_model=qwen-plus"


def test_sdk_shaped_v2_commands_are_not_rewritten():
    for argv in (
        ["apps", "completions", "create", "--app-id", "app-1"],
        ["multimodal", "chat", "create", "-m", "qwen-vl-plus"],
        ["rl", "jobs", "get", "job-1"],
        ["rl", "functions", "logs", "instance-1", "--keyword", "error"],
    ):
        assert normalize_argv(argv) == argv


def test_value_options_on_shared_commands_keep_their_values():
    assert normalize_argv(["models", "list", "--provider", "dashscope"]) == [
        "models",
        "list",
        "--provider",
        "dashscope",
    ]
    assert normalize_argv(["rl", "get", "job-1", "--output-format", "json"])[
        :2
    ] == ["--output", "json"]


def test_merging_legacy_fields_requires_an_inline_json_object():
    with pytest.raises(CliError) as caught:
        normalize_argv(
            [
                "tokenization",
                "create",
                "--parameters",
                "@params.json",
                "--enable-search",
            ],
        )
    assert caught.value.category == "usage"
