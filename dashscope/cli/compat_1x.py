# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Normalize compatible 1.x spellings on commands shared with v2.

The entry point routes legacy-only groups and dotted commands to the frozen
1.x implementation before calling this module. Shared commands are rewritten
here and executed by the v2 commands and SDK 2.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dashscope.cli.errors import usage_error


_DOTTED_COMMANDS = {
    "fine_tunes.call": ("ft", "create"),
    "fine_tunes.get": ("ft", "get"),
    "fine_tunes.list": ("ft", "list"),
    "fine_tunes.stream": ("ft", "stream"),
    "fine_tunes.cancel": ("ft", "cancel"),
    "fine_tunes.delete": ("ft", "delete"),
    "generation.call": ("generation", "create"),
    "files.upload": ("files", "upload"),
    "files.get": ("files", "get"),
    "files.list": ("files", "list"),
    "files.delete": ("files", "delete"),
    "deployments.call": ("deployments", "create"),
    "deployments.get": ("deployments", "get"),
    "deployments.list": ("deployments", "list"),
    "deployments.scale": ("deployments", "scale"),
    "deployments.delete": ("deployments", "delete"),
    "oss.upload": ("oss", "upload"),
}

_TOP_LEVEL_AUTH_COMMANDS = {"login", "whoami", "logout"}

_GROUPS = {
    "auth",
    "generation",
    "ft",
    "fine-tunes",
    "files",
    "deployments",
    "oss",
    "rerank",
    "embeddings",
    "tokenization",
    "models",
    "application",
    "image-synthesis",
    "video-synthesis",
    "image-generation",
    "multimodal-conversation",
    "multimodal-embedding",
    "transcription",
    "speech-synthesis",
    "rl",
    "agentic-rl",
}

_GROUP_ALIASES = {
    "fine_tunes": "ft",
    "fine-tunes": "ft",
    "agentic-rl": "rl",
}

_GROUP_HELP_TARGETS = {
    "application": "apps",
    "generation": "chat",
    "image-generation": "images",
    "image-synthesis": "images",
    "multimodal-conversation": "multimodal",
    "multimodal-embedding": "multimodal",
    "speech-synthesis": "audio",
    "transcription": "audio",
    "video-synthesis": "videos",
}

_ROOT_VALUE_OPTIONS = {
    "-k",
    "--api-key",
    "--api_key",
    "--protocol",
    "-u",
    "--base-url",
    "--base_url",
    "-w",
    "--workspace",
    "--region",
    "--timeout",
    "-o",
    "--output",
    "-c",
}

_ROOT_ALIASES = {
    "--api_key": "--api-key",
    "--base_url": "--base-url",
}

_OPTION_ALIASES = {
    **_ROOT_ALIASES,
    "--training_file_ids": "--training-file-ids",
    "--validation_file_ids": "--validation-file-ids",
    "--n_epochs": "--n-epochs",
    "--batch_size": "--batch-size",
    "--learning_rate": "--learning-rate",
    "--prompt_loss": "--prompt-loss",
    "--hyper_parameters": "--hyper-parameters",
    "--file_id": "--file-id",
    "--deployed_model": "--deployed-model",
    "--start_page": "--start-page",
    "--page_size": "--page-size",
}

_VALUE_OPTIONS = frozenset(
    {
        *_ROOT_VALUE_OPTIONS,
        "-a",
        "-b",
        "-d",
        "-e",
        "-f",
        "-i",
        "-l",
        "-m",
        "-n",
        "-p",
        "-q",
        "-s",
        "-t",
        "-v",
        "--app-id",
        "--model",
        "--prompt",
        "--messages",
        "--system",
        "--tools",
        "--temperature",
        "--top-p",
        "--top-k",
        "--max-tokens",
        "--stop",
        "--seed",
        "--repetition-penalty",
        "--presence-penalty",
        "--enable-search",
        "--n",
        "--result-format",
        "--history",
        "--parameters",
        "--session-id",
        "--memory-id",
        "--biz-params",
        "--image",
        "--file",
        "--rag-options",
        "--doc-tag-code",
        "--doc-reference-type",
        "--negative-prompt",
        "--size",
        "--page",
        "--page-size",
        "--start-page",
        "--image-url",
        "--function",
        "--style",
        "--extra-input",
        "--submit-timeout",
        "--model-name",
        "--api-key-id",
        "--status",
        "--start-time",
        "--end-time",
        "--img-url",
        "--first-frame-url",
        "--last-frame-url",
        "--duration",
        "--resolution",
        "--ratio",
        "--media",
        "--file-url",
        "--phrase-id",
        "--channel-id",
        "--speaker-count",
        "--special-word-filter",
        "--audio-format",
        "--format",
        "--sample-rate",
        "--voice",
        "--url",
        "--volume",
        "--rate",
        "--pitch",
        "--language-type",
        "--training-file-ids",
        "--validation-file-ids",
        "--mode",
        "--hyper-parameters",
        "--n-epochs",
        "--batch-size",
        "--learning-rate",
        "--prompt-loss",
        "--finetuned-model",
        "--purpose",
        "--description",
        "--file-id",
        "--deployed-model",
        "--capacity",
        "--model-version",
        "--suffix",
        "--plan",
        "--template-id",
        "--deploy-spec",
        "--billing-method",
        "--input",
        "--text-type",
        "--dimension",
        "--output-type",
        "--output-format",
        "--provider",
        "--instruct",
        "--text",
        "--audio",
        "--video",
        "--fps",
        "--res-level",
        "--max-video-frames",
        "--query",
        "--documents",
        "--document",
        "--top-n",
        "--rollout-classpaths",
        "--reward-classpaths",
        "--group-reward-classpaths",
        "--rollout",
        "--reward",
        "--group-reward",
        "--workspace-dir",
        "--training-files",
        "--validation-files",
        "--training-file",
        "--validation-file",
        "--teacher-model",
        "--job-name",
        "--set",
        "--instance-id",
        "--type",
        "--offset",
        "--lines",
        "--config",
        "--key",
    },
)

_COMMON_FLAGS = frozenset(
    {
        "-h",
        "--help",
        "--stream",
        "--yes",
        "-y",
        "--wait",
        "--no-wait",
        "--verbose",
        "--has-thoughts",
        "--no-has-thoughts",
        "--enable-thinking",
        "--no-enable-thinking",
        "--no-enable-search",
        "--enable-fusion",
        "--disable-fusion",
        "--no-enable-fusion",
        "--return-documents",
        "--no-return-documents",
        "--prompt-extend",
        "--no-prompt-extend",
        "--watermark",
        "--no-watermark",
        "--lazy-load",
        "--no-lazy-load",
        "--pull-logs",
        "--all",
        "--enable-ssml",
        "--no-enable-ssml",
        "--disfluency-removal-enabled",
        "--diarization-enabled",
        "--timestamp-alignment-enabled",
        "--audio-event-detection-enabled",
    },
)

_HOISTED_OPTIONS = {
    "-k": "--api-key",
    "--api-key": "--api-key",
    "-w": "--workspace",
    "--workspace": "--workspace",
    "-u": "--base-url",
    "--base-url": "--base-url",
}


@dataclass
class _Parsed:
    options: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    positionals: List[str] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return any(name in names for name, _ in self.options)

    def take(self, *names: str) -> List[Tuple[str, Optional[str]]]:
        taken = [item for item in self.options if item[0] in names]
        self.options = [item for item in self.options if item[0] not in names]
        return taken

    def values(self, *names: str) -> List[Optional[str]]:
        return [value for _, value in self.take(*names)]

    def add(self, name: str, value: Optional[str] = None) -> None:
        self.options.append((name, value))

    def add_flag(self, name: str) -> None:
        if not self.has(name):
            self.add(name)

    def rename(self, mapping: Dict[str, str]) -> None:
        self.options = [
            (mapping.get(name, name), value) for name, value in self.options
        ]

    def emit(self) -> List[str]:
        result: List[str] = []
        for name, value in self.options:
            result.append(name)
            if value is not None:
                result.append(value)
        result.extend(self.positionals)
        return result


def _split_option(token: str) -> Tuple[str, Optional[str]]:
    name, separator, value = token.partition("=")
    alias = _OPTION_ALIASES.get(name)
    if alias is None and name.startswith("--"):
        alias = name.replace("_", "-")
    alias = alias or name
    return alias, value if separator else None


def _normalise_token(token: str) -> str:
    name, value = _split_option(token)
    return f"{name}={value}" if value is not None else name


def _command_index(argv: Sequence[str]) -> Optional[int]:
    index = 0
    while index < len(argv):
        token = argv[index]
        name, value = _split_option(token)
        if name in _ROOT_VALUE_OPTIONS:
            index += 1 if value is not None else 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return index
    return None


def _parse(rest: Sequence[str], flags: Iterable[str]) -> _Parsed:
    parsed = _Parsed()
    flags = _COMMON_FLAGS | frozenset(flags)
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--":
            parsed.positionals.extend(rest[index + 1 :])
            break
        if not token.startswith("-"):
            parsed.positionals.append(token)
            index += 1
            continue
        name, value = _split_option(token)
        if (
            value is None
            and name not in flags
            and name in _VALUE_OPTIONS
            and index + 1 < len(rest)
        ):
            value = rest[index + 1]
            index += 1
        parsed.add(name, value)
        index += 1
    return parsed


def _json_scalar(raw: Optional[str]) -> object:
    if raw is None:
        return True
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _boolean(raw: Optional[str], option: str) -> bool:
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise usage_error(f"{option} expects true or false, got {raw!r}")


def _merge_json_option(
    parsed: _Parsed,
    option: str,
    values: Dict[str, object],
) -> None:
    merged: Dict[str, object] = {}
    existing = parsed.values(option)
    for raw in existing:
        try:
            value = json.loads(raw or "")
        except json.JSONDecodeError as exc:
            raise usage_error(
                f"{option} must be an inline JSON object when combined with "
                "a 1.x option",
            ) from exc
        if not isinstance(value, dict):
            raise usage_error(f"{option} must contain a JSON object")
        merged.update(value)
    merged.update(values)
    if merged or existing:
        parsed.add(option, json.dumps(merged, ensure_ascii=False))


def _one(values: List[Optional[str]]) -> Optional[str]:
    return values[-1] if values else None


def _bool_value_flag(parsed: _Parsed, option: str, negative: str) -> None:
    for raw in parsed.values(option):
        parsed.add(option if _boolean(raw, option) else negative)


def _hoist_globals(
    prefix: List[str],
    parsed: _Parsed,
    *,
    auth_login: bool,
    rl: bool,
    speech: bool,
) -> List[str]:
    hoisted: List[str] = []
    mapping = dict(_HOISTED_OPTIONS)
    if speech:
        mapping["--url"] = "--base-url"
    if rl:
        mapping.update(
            {
                "-o": "--output",
                "--output-format": "--output",
            },
        )
    for name, value in parsed.take(*mapping):
        if auth_login and name == "-k":
            parsed.add(name, value)
            continue
        target = mapping[name]
        hoisted.append(target)
        if value is not None:
            hoisted.append(value)
    if rl:
        for name, _ in parsed.take("--verbose"):
            hoisted.append(name)
        if parsed.has("-v"):
            parsed.take("-v")
            hoisted.append("--verbose")
    return prefix + hoisted


def _normalize_generation(parsed: _Parsed) -> None:
    parsed.rename({"-p": "--prompt", "-s": "--stream"})
    stops = parsed.values("--stop")
    for raw in stops:
        for stop in (raw or "").split(","):
            if stop.strip():
                parsed.add("--stop", stop.strip())
    _bool_value_flag(parsed, "--enable-search", "--no-enable-search")
    result_format = _one(parsed.values("--result-format"))
    if result_format is not None:
        _merge_json_option(
            parsed,
            "--parameters",
            {"result_format": result_format},
        )
    parsed.take("--history")


def _normalize_multimodal_conversation(parsed: _Parsed) -> None:
    text = _one(parsed.values("-t", "--text"))
    images = parsed.values("--image")
    content = [{"image": image} for image in images if image is not None]
    if text is not None:
        content.append({"text": text})
    parsed.add(
        "--messages",
        json.dumps(
            [{"role": "user", "content": content}],
            ensure_ascii=False,
        ),
    )
    result_format = _one(parsed.values("--result-format"))
    if result_format is not None:
        _merge_json_option(
            parsed,
            "--parameters",
            {"result_format": result_format},
        )


def _normalize_image_generation(parsed: _Parsed, subcommand: str) -> None:
    if subcommand != "create":
        return
    parsed.rename(
        {"-t": "--prompt", "--text": "--prompt", "--image": "--image-url"},
    )
    max_images = _one(parsed.values("--max-images"))
    if max_images is not None:
        _merge_json_option(
            parsed,
            "--parameters",
            {"max_images": _json_scalar(max_images)},
        )


def _normalize_video(parsed: _Parsed, subcommand: str) -> None:
    if subcommand != "create":
        return
    parsed.rename({"--img-url": "--image-url"})
    extra: Dict[str, object] = {"extend_prompt": True}
    for option, key in (
        ("--negative-prompt", "negative_prompt"),
        ("--first-frame-url", "first_frame_url"),
        ("--last-frame-url", "last_frame_url"),
    ):
        value = _one(parsed.values(option))
        if value is not None:
            extra[key] = value
    _merge_json_option(parsed, "--extra-input", extra)


def _normalize_transcription(parsed: _Parsed, subcommand: str) -> None:
    if subcommand != "create":
        return
    urls = [
        value for value in parsed.values("--file-url") if value is not None
    ]
    parsed.positionals = urls + parsed.positionals
    parsed.add_flag("--no-wait")
    channels = [_json_scalar(value) for value in parsed.values("--channel-id")]
    parameters: Dict[str, object] = {}
    if channels:
        parameters["channel_id"] = channels
    for option in (
        "--disfluency-removal-enabled",
        "--diarization-enabled",
        "--timestamp-alignment-enabled",
        "--audio-event-detection-enabled",
    ):
        if parsed.has(option):
            parsed.take(option)
            parameters[option[2:].replace("-", "_")] = True
    for option in ("--speaker-count", "--special-word-filter"):
        value = _one(parsed.values(option))
        if value is not None:
            parameters[option[2:].replace("-", "_")] = _json_scalar(value)
    _merge_json_option(parsed, "--parameters", parameters)


def _normalize_speech(parsed: _Parsed) -> None:
    text = _one(parsed.values("-t", "--text"))
    if text is not None:
        parsed.positionals.insert(0, text)
    parsed.rename({"--audio-format": "--format"})
    if not parsed.has("--format"):
        parsed.add("--format", "wav")
    if not parsed.has("--sample-rate"):
        parsed.add("--sample-rate", "24000")
    extra: Dict[str, object] = {}
    for option in ("--volume", "--rate", "--pitch"):
        value = _one(parsed.values(option))
        if value is not None:
            extra[option[2:]] = _json_scalar(value)
    _merge_json_option(parsed, "--extra-input", extra)


def _normalize_ft(parsed: _Parsed, subcommand: str) -> None:
    if subcommand == "create":
        parsed.add_flag("--wait")
    if subcommand == "list":
        parsed.rename(
            {
                "-p": "--page",
                "-s": "--size",
                "--page-size": "--size",
                "--start-page": "--page",
            },
        )
    if subcommand in {"cancel", "delete"}:
        parsed.add_flag("--yes")


def _normalize_files(parsed: _Parsed, subcommand: str) -> None:
    if subcommand == "upload":
        parsed.rename({"-p": "--purpose", "-d": "--description"})
        file_path = _one(parsed.values("-f", "--file"))
        if file_path is not None:
            parsed.positionals.insert(0, file_path)
    if subcommand in {"get", "delete"}:
        file_id = _one(parsed.values("--file-id"))
        if file_id is not None:
            parsed.positionals.insert(0, file_id)
    if subcommand == "list":
        parsed.rename(
            {
                "-p": "--page",
                "-s": "--size",
                "--page-size": "--size",
                "--start-page": "--page",
            },
        )
    if subcommand == "delete":
        parsed.add_flag("--yes")


def _normalize_deployments(parsed: _Parsed, subcommand: str) -> None:
    parsed.rename({"-c": "--capacity"})
    if subcommand in {"call", "create"}:
        parsed.rename({"-s": "--suffix"})
        parsed.add_flag("--wait")
    if subcommand == "list":
        parsed.rename(
            {
                "-p": "--page",
                "-s": "--size",
                "--page-size": "--size",
                "--start-page": "--page",
            },
        )
    if subcommand in {"get", "scale", "delete"}:
        deployed_model = _one(parsed.values("--deployed-model"))
        if deployed_model is not None:
            parsed.positionals.insert(0, deployed_model)
    if subcommand in {"scale", "delete"}:
        parsed.add_flag("--yes")


def _normalize_models(parsed: _Parsed, subcommand: str) -> None:
    if subcommand == "list":
        parsed.rename(
            {
                "-p": "--page",
                "-s": "--size",
                "--page-size": "--size",
                "--start-page": "--page",
            },
        )


def _normalize_embeddings(parsed: _Parsed, subcommand: str) -> None:
    if subcommand != "create":
        return
    parsed.rename({"-i": "--input", "-t": "--text-type"})
    inputs = [value for value in parsed.values("--input") if value is not None]
    if len(inputs) == 1:
        parsed.add("--input", inputs[0])
    elif inputs:
        parsed.add("--input", json.dumps(inputs, ensure_ascii=False))


def _normalize_rl(parsed: _Parsed, subcommand: str) -> None:
    if subcommand == "register-functions":
        parsed.rename(
            {
                "--rollout-classpaths": "--rollout",
                "--reward-classpaths": "--reward",
                "--group-reward-classpaths": "--group-reward",
            },
        )
    elif subcommand == "test-functions":
        parsed.rename({"-t": "--type", "-i": "--input"})
        if parsed.positionals:
            parsed.add("--instance-id", parsed.positionals.pop(0))
    elif subcommand == "upload-data":
        parsed.rename(
            {
                "--training-files": "--training-file",
                "--validation-files": "--validation-file",
            },
        )
    elif subcommand == "run":
        parsed.rename({"-c": "--config"})
        teacher_model = _one(parsed.values("--teacher-model"))
        if teacher_model is not None:
            parsed.add("--set", f"teacher_model={teacher_model}")
    elif subcommand == "list":
        parsed.rename({"-p": "--page", "-s": "--size"})
    if subcommand == "cancel":
        parsed.add_flag("--yes")


def _flags_for(group: str, subcommand: str) -> frozenset:
    flags = set()
    if group == "generation":
        flags.update({"-s"})
    if group == "tokenization":
        flags.add("--enable-search")
    if group == "rl" and subcommand == "run":
        flags.add("-v")
    return frozenset(flags)


# pylint: disable-next=too-many-return-statements
def _target(group: str, subcommand: str) -> Tuple[str, List[str], str]:
    if group == "generation":
        return "chat", ["create"], "create"
    if group == "application":
        return "apps", ["call"], "call"
    if group in {"image-generation", "image-synthesis"}:
        verbs = {"fetch": "get"}
        verb = verbs.get(subcommand, subcommand)
        return "images", ["jobs", verb], verb
    if group == "video-synthesis":
        verbs = {"fetch": "get"}
        verb = verbs.get(subcommand, subcommand)
        return "videos", ["jobs", verb], verb
    if group == "multimodal-conversation":
        return "multimodal", ["create"], "create"
    if group == "multimodal-embedding":
        return "multimodal", ["embeddings", "create"], "create"
    if group == "transcription":
        verbs = {"create": "transcribe", "fetch": "get"}
        verb = verbs.get(subcommand, subcommand)
        return "audio", ["asr", verb], verb
    if group == "speech-synthesis":
        return "audio", ["tts", "http"], "http"
    if group == "tokenization":
        return "tokenization", ["count"], "count"
    verb = "create" if subcommand == "call" else subcommand
    if group in {"rl", "agentic-rl"}:
        verb = verb.replace("_", "-")
        function_verbs = {
            "register-functions": "register",
            "test-functions": "test",
            "upload-data": "upload-data",
        }
        if verb in function_verbs:
            return "rl", ["functions", function_verbs[verb]], verb
        if verb == "run":
            return "rl", ["jobs", "create"], verb
        return "rl", ["jobs", verb], verb
    return _GROUP_ALIASES.get(group, group), [verb], verb


# pylint: disable-next=too-many-branches,too-many-statements
def normalize_argv(argv: Sequence[str]) -> List[str]:
    """Return *argv* in v2 form, accepting every registered 1.x command."""
    args = list(argv)
    index = _command_index(args)
    if index is None:
        return [_normalise_token(arg) for arg in args]
    prefix = [_normalise_token(arg) for arg in args[:index]]
    token = args[index]
    if token == "expert" or (
        token not in _GROUPS
        and token not in _DOTTED_COMMANDS
        and token not in _TOP_LEVEL_AUTH_COMMANDS
    ):
        return prefix + args[index:]
    if token in _TOP_LEVEL_AUTH_COMMANDS:
        group, subcommand = "auth", token
        rest = args[index + 1 :]
    elif token in _DOTTED_COMMANDS:
        group, subcommand = _DOTTED_COMMANDS[token]
        rest = args[index + 1 :]
    else:
        group = token
        rest = args[index + 1 :]
        if (
            group in {"rl", "agentic-rl"}
            and rest
            and rest[0] in {"jobs", "functions"}
        ):
            return prefix + ["rl"] + rest
        subcommand = ""
        if rest and not rest[0].startswith("-"):
            subcommand = rest.pop(0)
    if not subcommand:
        target = _GROUP_HELP_TARGETS.get(
            group,
            _GROUP_ALIASES.get(group, group),
        )
        return prefix + [target] + rest

    original_subcommand = subcommand
    target_group, target_path, normalized_subcommand = _target(
        group,
        subcommand,
    )
    parsed = _parse(rest, _flags_for(group, original_subcommand))
    root = _hoist_globals(
        prefix,
        parsed,
        auth_login=group == "auth" and original_subcommand == "login",
        rl=group in {"rl", "agentic-rl"},
        speech=group == "speech-synthesis",
    )

    if group == "generation":
        _normalize_generation(parsed)
    elif group == "application":
        parsed.rename({"-a": "--app-id", "-p": "--prompt"})
    elif group == "multimodal-conversation":
        _normalize_multimodal_conversation(parsed)
    elif group == "multimodal-embedding":
        parsed.rename({"--disable-fusion": "--no-enable-fusion"})
    elif group in {"image-generation", "image-synthesis"}:
        _normalize_image_generation(parsed, normalized_subcommand)
    elif group == "video-synthesis":
        _normalize_video(parsed, normalized_subcommand)
    elif group == "transcription":
        _normalize_transcription(parsed, original_subcommand)
    elif group == "speech-synthesis":
        _normalize_speech(parsed)
    elif group in {"ft", "fine-tunes"}:
        _normalize_ft(parsed, normalized_subcommand)
    elif group == "files":
        _normalize_files(parsed, original_subcommand)
    elif group == "deployments":
        _normalize_deployments(parsed, original_subcommand)
    elif group == "models":
        _normalize_models(parsed, original_subcommand)
    elif group == "embeddings":
        _normalize_embeddings(parsed, original_subcommand)
    elif group == "rerank":
        parsed.rename({"-n": "--top-n", "-i": "--instruct"})
    elif group == "tokenization":
        enable_search = parsed.has("--enable-search")
        parsed.take("--enable-search")
        if enable_search:
            _merge_json_option(
                parsed,
                "--parameters",
                {"enable_search": True},
            )
        parsed.rename({"-p": "--prompt"})
    elif group in {"rl", "agentic-rl"}:
        _normalize_rl(parsed, normalized_subcommand)

    if group in {"image-generation", "image-synthesis", "video-synthesis"}:
        parsed.rename(
            {
                "--model-name": "--model",
                "-p": "--page",
                "-s": "--size",
                "--page-size": "--size",
            },
        )
        if normalized_subcommand == "cancel":
            parsed.add_flag("--yes")
    if group == "transcription":
        parsed.rename({"-p": "--page", "-s": "--size"})
        if normalized_subcommand == "cancel":
            parsed.add_flag("--yes")

    return root + [target_group] + target_path + parsed.emit()
