# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import pytest

from dashscope import __version__ as sdk_version
from dashscope.acli import SDK_SESSION_ID as ACLI_SESSION_ID
from dashscope.acli import __version__ as acli_version
from dashscope.acli.providers.tongyi import TongyiProvider
from dashscope.common.utils import _SDK_SESSION_ID, get_sdk_headers

CLIENT_HEADER = "x-dashscope-sdk-client"
SESSION_HEADER = "x-dashscope-sdk-session-id"
DISABLE_ENV = "DASHSCOPE_DISABLE_SDK_HEADERS"

# client value sources:
#   - "python-sdk": SDK default (_SDK_CLIENT in common/utils.py)
#   - "python-cli": dashscope/cli/__init__.py calls
#                   set_sdk_client("python-cli") at CLI process startup
#   - "acli":       acli/providers/tongyi.py builds its own header
# module rules (see get_api_module in common/utils.py):
#   - service calls (BaseApi/BaseAioApi/BaseAsyncApi/BaseAsyncAioApi.call):
#     first package segment of the API class, e.g. aigc/audio/embeddings/
#     nlp/rerank/app/tokenizers/assistants(threads pkg)/finetune/models/files
#   - task management (fetch/cancel/wait/list on /tasks/* endpoints): "tasks"
#   - agentstudio: passed explicitly by agentstudio/transport.py
#   - realtime/dialog WebSocket: passed explicitly where headers are built
#   - acli side: TongyiProvider ctor arg, defaults to "app"
SDK_MODULES = ["", "agentstudio"]
ACLI_MODULES = ["", "app"]


@pytest.fixture(autouse=True)
def _pin_sdk_client(monkeypatch):
    # CLI tests switch the client to python-cli in-process; pin the default
    monkeypatch.setattr("dashscope.common.utils._SDK_CLIENT", "python-sdk")


def _check_client_header(value, client, version, module=""):
    parts = value.split("/")
    assert parts[0] == client
    assert parts[1] == version
    if module:
        assert len(parts) == 3
        assert parts[2] == module
    else:
        assert len(parts) == 2


def test_python_sdk_client_header():
    print("\n=== python-sdk: x-dashscope-sdk-client ===")
    for module in SDK_MODULES:
        headers = get_sdk_headers(module=module)
        value = headers[CLIENT_HEADER]
        print(f"module={module or '(none)':<15} -> {value}")
        _check_client_header(value, "python-sdk", sdk_version, module)
        assert headers[SESSION_HEADER] == _SDK_SESSION_ID


def test_acli_client_header():
    # pylint: disable=protected-access
    print("\n=== acli: x-dashscope-sdk-client ===")
    default = TongyiProvider(model="qwen-plus", api_key="sk-x")
    value = default._get_headers()[CLIENT_HEADER]
    print(f"module=(default app)   -> {value}")
    _check_client_header(value, "acli", acli_version, "app")
    for module in ACLI_MODULES:
        provider = TongyiProvider(
            model="qwen-plus",
            api_key="sk-x",
            module=module,
        )
        headers = provider._get_headers()
        value = headers[CLIENT_HEADER]
        print(f"module={module or '(none)':<15} -> {value}")
        _check_client_header(value, "acli", acli_version, module)
        assert headers[SESSION_HEADER] == ACLI_SESSION_ID


def test_sdk_headers_disabled(monkeypatch):
    # pylint: disable=protected-access
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert not get_sdk_headers()
    assert not get_sdk_headers(module="agentstudio")
    provider = TongyiProvider(model="qwen-plus", api_key="sk-x", module="app")
    assert CLIENT_HEADER not in provider._get_headers()
    assert SESSION_HEADER not in provider._get_headers()


def test_python_cli_client_header():
    from dashscope.common.utils import set_sdk_client

    set_sdk_client("python-cli")
    try:
        headers = get_sdk_headers()
        value = headers[CLIENT_HEADER]
        print(f"\npython-cli -> {value}")
        _check_client_header(value, "python-cli", sdk_version)
        # the module segment applies as well
        value = get_sdk_headers(module="agentstudio")[CLIENT_HEADER]
        print(f"python-cli -> {value}")
        _check_client_header(value, "python-cli", sdk_version, "agentstudio")
    finally:
        set_sdk_client("python-sdk")


def test_cli_import_marks_process():
    # subprocess isolation check: after importing dashscope.cli, requests in
    # that process must carry client=python-cli (independent of test order)
    import subprocess
    import sys

    code = (
        "import dashscope.cli; "
        "from dashscope.common.utils import get_sdk_headers; "
        "h = get_sdk_headers(); "
        "c = h['x-dashscope-sdk-client']; "
        "assert c.startswith('python-cli/'), c; "
        "print(c)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    print(f"\nCLI process -> {result.stdout.strip()}")


# ---------------------------------------------------------------------------
# module segment: full API coverage checks
# ---------------------------------------------------------------------------


def test_get_api_module_mapping():
    from dashscope.common.utils import get_api_module

    cases = {
        "dashscope.aigc.generation": "aigc",
        "dashscope.aigc.multimodal_conversation": "aigc",
        "dashscope.audio.tts.speech_synthesizer": "audio",
        "dashscope.audio.asr.recognition": "audio",
        "dashscope.embeddings.text_embedding": "embeddings",
        "dashscope.nlp.understanding": "nlp",
        "dashscope.rerank.text_rerank": "rerank",
        "dashscope.app.application": "app",
        "dashscope.tokenizers.tokenization": "tokenizers",
        "dashscope.assistants.assistants": "assistants",
        # threads/messages/runs/steps are grouped into the assistants module
        "dashscope.threads.threads": "assistants",
        "dashscope.threads.runs.runs": "assistants",
        "dashscope.models": "models",
        "dashscope.files": "files",
        "dashscope.finetune.finetunes": "finetune",
        "dashscope.multimodal.tingwu.tingwu": "multimodal",
        "dashscope.common.utils": "common",  # non-API classes never reach this
        "other.lib": "",
    }
    for module_name, expected in cases.items():
        assert get_api_module(module_name) == expected, module_name


class _CapturedRequest(Exception):
    """Sentinel: transport layer captured the outgoing request headers."""

    def __init__(self, headers):
        super().__init__("captured")
        self.headers = headers or {}


@pytest.fixture
def capture_headers(monkeypatch):
    """Patch all transport boundaries to capture headers without network."""
    import aiohttp
    import requests

    from dashscope.api_entities.http_request import HttpRequest
    from dashscope.api_entities.websocket_request import WebSocketRequest

    def _raise(headers):
        raise _CapturedRequest(headers)

    def http_call(self):
        _raise(self.headers)

    async def http_aio_call(self):
        _raise(self.headers)

    def ws_call(self):
        _raise(self.headers)

    monkeypatch.setattr(HttpRequest, "call", http_call)
    monkeypatch.setattr(HttpRequest, "aio_call", http_aio_call)
    monkeypatch.setattr(WebSocketRequest, "call", ws_call)

    def session_request(self, *args, **kwargs):
        _raise(kwargs.get("headers"))

    for method in ("get", "post", "delete", "put", "patch"):
        monkeypatch.setattr(requests.Session, method, session_request)

    def aio_get(self, *args, **kwargs):
        _raise(kwargs.get("headers"))

    monkeypatch.setattr(aiohttp.ClientSession, "get", aio_get)

    monkeypatch.setattr("dashscope.api_key", "sk-test")
    return None


def _capture(fn):
    try:
        fn()
    except _CapturedRequest as e:
        return e.headers
    raise AssertionError("no request was issued")


def _check_module(headers, module, label):
    value = headers.get(CLIENT_HEADER)
    assert value, f"{label}: missing {CLIENT_HEADER} in {headers}"
    expected = f"python-sdk/{sdk_version}/{module}"
    print(f"{label:<45} -> {value}")
    assert value == expected, f"{label}: {value} != {expected}"


def test_service_api_modules(capture_headers, tmp_path):
    # pylint: disable=unused-argument,import-outside-toplevel
    import asyncio

    from dashscope import (
        Application,
        CodeGeneration,
        Conversation,
        Generation,
        ImageSynthesis,
        MultiModalConversation,
        MultiModalEmbedding,
        TextEmbedding,
        TextReRank,
        Tokenization,
        Understanding,
        VideoSynthesis,
    )
    from dashscope.aigc.generation import AioGeneration
    from dashscope.aigc.multimodal_conversation import (
        AioMultiModalConversation,
    )
    from dashscope.audio.asr.recognition import (
        Recognition,
        RecognitionCallback,
    )
    from dashscope.audio.asr.transcription import Transcription
    from dashscope.audio.asr.translation_recognizer import (
        TranslationRecognizerCallback,
        TranslationRecognizerRealtime,
    )
    from dashscope.audio.http_tts.http_speech_synthesizer import (
        HttpSpeechSynthesizer,
    )
    from dashscope.audio.qwen_asr.qwen_transcription import QwenTranscription
    from dashscope.audio.tts.speech_synthesizer import SpeechSynthesizer
    from dashscope.embeddings.batch_text_embedding import BatchTextEmbedding
    from dashscope.embeddings.multimodal_embedding import (
        AioMultiModalEmbedding,
    )
    from dashscope.rerank.text_rerank import AioTextReRank

    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"\x00" * 128)

    cases = [
        # aigc/
        ("aigc: Generation.call",
         lambda: Generation.call(model="m", prompt="hi"), "aigc"),
        ("aigc: AioGeneration.call",
         lambda: asyncio.run(AioGeneration.call(model="m", prompt="hi")),
         "aigc"),
        ("aigc: Conversation.call",
         lambda: Conversation().call(model="m", prompt="hi"), "aigc"),
        ("aigc: CodeGeneration.call",
         lambda: CodeGeneration.call(
             model="m",
             scene="custom",
             message=[{"role": "user", "content": "hi"}],
         ), "aigc"),
        ("aigc: ImageSynthesis.async_call",
         lambda: ImageSynthesis.async_call(model="m", prompt="a cat"),
         "aigc"),
        ("aigc: MultiModalConversation.call",
         lambda: MultiModalConversation.call(
             model="m",
             messages=[{"role": "user", "content": [{"text": "hi"}]}],
         ), "aigc"),
        ("aigc: AioMultiModalConversation.call",
         lambda: asyncio.run(AioMultiModalConversation.call(
             model="m",
             messages=[{"role": "user", "content": [{"text": "hi"}]}],
         )), "aigc"),
        ("aigc: VideoSynthesis.async_call",
         lambda: VideoSynthesis.async_call(model="m", prompt="a cat"),
         "aigc"),
        # audio/
        ("audio: SpeechSynthesizer.call (ws)",
         lambda: SpeechSynthesizer.call(model="m", text="hi"), "audio"),
        ("audio: HttpSpeechSynthesizer.call",
         lambda: HttpSpeechSynthesizer.call(model="m", text="hi", voice="v"),
         "audio"),
        ("audio: Transcription.async_call",
         lambda: Transcription.async_call(
             model="m", file_urls=["https://example.com/a.wav"],
         ), "audio"),
        ("audio: Recognition.call (ws)",
         lambda: Recognition(
             model="m", format="wav", sample_rate=16000,
             callback=RecognitionCallback(),
         ).call(file=str(audio_file)), "audio"),
        ("audio: TranslationRecognizerRealtime.call (ws)",
         lambda: TranslationRecognizerRealtime(
             model="m", format="wav", sample_rate=16000,
             callback=TranslationRecognizerCallback(),
         ).call(file=str(audio_file)), "audio"),
        ("audio: QwenTranscription.async_call",
         lambda: QwenTranscription.async_call(
             model="m", file_url="https://example.com/a.wav",
         ), "audio"),
        # embeddings/
        ("embeddings: TextEmbedding.call",
         lambda: TextEmbedding.call(model="m", input="hi"), "embeddings"),
        ("embeddings: BatchTextEmbedding.call",
         lambda: BatchTextEmbedding.call(
             model="m", url="https://example.com/input.txt",
         ), "embeddings"),
        ("embeddings: MultiModalEmbedding.call",
         lambda: MultiModalEmbedding.call(
             model="m", input=[{"text": "hi"}],
         ), "embeddings"),
        ("embeddings: AioMultiModalEmbedding.call",
         lambda: asyncio.run(AioMultiModalEmbedding.call(
             model="m", input=[{"text": "hi"}],
         )), "embeddings"),
        # nlp/
        ("nlp: Understanding.call",
         lambda: Understanding.call(model="m", sentence="hi", labels="lbl"),
         "nlp"),
        # rerank/
        ("rerank: TextReRank.call",
         lambda: TextReRank.call(model="m", query="q", documents=["d"]),
         "rerank"),
        ("rerank: AioTextReRank.call",
         lambda: asyncio.run(AioTextReRank.call(
             model="m", query="q", documents=["d"],
         )), "rerank"),
        # app/
        ("app: Application.call",
         lambda: Application.call(app_id="app1", prompt="hi"), "app"),
        # tokenizers/
        ("tokenizers: Tokenization.call",
         lambda: Tokenization.call(model="m", prompt="hi"), "tokenizers"),
    ]
    print("\n=== service calls: python-sdk/<version>/<module> ===")
    for label, fn, module in cases:
        _check_module(_capture(fn), module, label)


def test_tasks_module(capture_headers):
    # pylint: disable=unused-argument,import-outside-toplevel
    import asyncio

    from dashscope import ImageSynthesis
    from dashscope.aigc.image_synthesis import AioImageSynthesis
    from dashscope.aigc.video_synthesis import VideoSynthesis

    cases = [
        ("tasks: ImageSynthesis.fetch",
         lambda: ImageSynthesis.fetch("task-1")),
        ("tasks: ImageSynthesis.cancel",
         lambda: ImageSynthesis.cancel("task-1")),
        ("tasks: ImageSynthesis.list", lambda: ImageSynthesis.list()),
        ("tasks: VideoSynthesis.fetch",
         lambda: VideoSynthesis.fetch("task-1")),
        ("tasks: AioImageSynthesis.fetch",
         lambda: asyncio.run(AioImageSynthesis.fetch("task-1"))),
        ("tasks: AioImageSynthesis.list",
         lambda: asyncio.run(AioImageSynthesis.list())),
    ]
    print("\n=== task management (/tasks/*): module=tasks ===")
    for label, fn in cases:
        _check_module(_capture(fn), "tasks", label)


def test_resource_api_modules(capture_headers):
    # pylint: disable=unused-argument,import-outside-toplevel
    from dashscope import (
        Assistants,
        Deployments,
        Files,
        FineTunes,
        Messages,
        Models,
        Runs,
        Steps,
        Threads,
    )

    cases = [
        ("models: Models.list", lambda: Models.list(), "models"),
        ("models: Models.get", lambda: Models.get("m"), "models"),
        ("files: Files.list", lambda: Files.list(), "files"),
        ("finetune: FineTunes.list", lambda: FineTunes.list(), "finetune"),
        ("finetune: FineTunes.cancel",
         lambda: FineTunes.cancel("job-1"), "finetune"),
        ("finetune: Deployments.list", lambda: Deployments.list(), "finetune"),
        ("assistants: Assistants.list", lambda: Assistants.list(),
         "assistants"),
        ("assistants: Threads.create", lambda: Threads.create(),
         "assistants"),
        ("assistants: Messages.list",
         lambda: Messages.list(thread_id="t1"), "assistants"),
        ("assistants: Runs.list", lambda: Runs.list(thread_id="t1"),
         "assistants"),
        ("assistants: Steps.list",
         lambda: Steps.list(thread_id="t1", run_id="r1"), "assistants"),
    ]
    print("\n=== resource management APIs ===")
    for label, fn, module in cases:
        _check_module(_capture(fn), module, label)


def test_realtime_websocket_modules():
    # pylint: disable=import-outside-toplevel
    import dashscope
    from dashscope.audio.qwen_omni.omni_realtime import (
        OmniRealtimeCallback,
        OmniRealtimeConversation,
    )
    from dashscope.audio.qwen_tts_realtime.qwen_tts_realtime import (
        QwenTtsRealtime,
    )
    from dashscope.audio.tts_v2.speech_synthesizer import Request as TtsV2Req
    from dashscope.multimodal.multimodal_dialog import (
        _Request as DialogRequest,
    )
    from dashscope.multimodal.tingwu.tingwu_realtime import (
        _Request as TingwuRequest,
    )

    dashscope.api_key = "sk-test"

    class _Cb(OmniRealtimeCallback):
        pass

    cases = [
        ("audio: tts_v2 SpeechSynthesizer",
         TtsV2Req(apikey="sk-test", model="m", voice="v")
         .get_websocket_headers(None, None), "audio"),
        ("audio: OmniRealtimeConversation",
         OmniRealtimeConversation(model="m", callback=_Cb())
         ._get_websocket_header(), "audio"),
        ("audio: QwenTtsRealtime",
         QwenTtsRealtime(model="m")._get_websocket_header(), "audio"),
        # MultiModalDialog is grouped into aigc per the mapping table
        ("aigc: MultiModalDialog",
         DialogRequest().get_websocket_header("sk-test"), "aigc"),
        ("multimodal: TingWuRealtime",
         TingwuRequest().get_websocket_header("sk-test"), "multimodal"),
    ]
    print("\n=== realtime/dialog WebSocket ===")
    for label, headers, module in cases:
        _check_module(headers, module, label)


def test_encryption_module(monkeypatch):
    # pylint: disable=import-outside-toplevel
    from dashscope.api_entities.encryption import Encryption

    def fake_get(url, headers=None, timeout=None):
        raise _CapturedRequest(headers)

    monkeypatch.setattr("dashscope.api_entities.encryption.requests.get",
                        fake_get)
    print("\n=== misc ===")
    try:
        Encryption._get_public_keys()  # pylint: disable=protected-access
    except _CapturedRequest as e:
        _check_module(e.headers, "utils", "utils: Encryption public-keys")
    else:
        raise AssertionError("no request was issued")


def test_finetune_rl_module(monkeypatch):
    # pylint: disable=import-outside-toplevel
    import asyncio

    import dashscope.finetune.reinforcement.common.utils as rl_utils

    captured = {}

    async def fake_async_http_request(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(rl_utils, "async_http_request",
                        fake_async_http_request)
    asyncio.run(
        rl_utils.client_fc(api_key="sk-test", url="http://x", request_data={})
    )
    _check_module(captured.get("headers"), "finetune",
                  "finetune: rl client_fc")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
