# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# pylint: disable=wrong-import-position,undefined-all-variable

import importlib
import logging
import warnings
from logging import NullHandler

# Suppress urllib3 NotOpenSSLWarning on systems with LibreSSL before any SDK
# submodule imports urllib3.
warnings.filterwarnings(
    "ignore",
    message=".*urllib3.*only supports OpenSSL.*",
    category=Warning,
)

from dashscope.legacy_1x._shim import install as _install_legacy_aliases

_install_legacy_aliases()

from dashscope.version import __version__
from dashscope.legacy_1x.aigc.code_generation import CodeGeneration
from dashscope.legacy_1x.aigc.conversation import (
    Conversation,
    History,
    HistoryItem,
)
from dashscope.legacy_1x.aigc.generation import AioGeneration, Generation
from dashscope.legacy_1x.aigc.image_synthesis import ImageSynthesis
from dashscope.legacy_1x.aigc.multimodal_conversation import (
    MultiModalConversation,
    AioMultiModalConversation,
)
from dashscope.legacy_1x.aigc.video_synthesis import VideoSynthesis
from dashscope.legacy_1x.app.application import Application
from dashscope.legacy_1x.audio.asr.transcription import Transcription
from dashscope.legacy_1x.audio.http_tts.http_speech_synthesizer import (
    HttpSpeechSynthesizer,
)
from dashscope.legacy_1x.audio.tts.speech_synthesizer import SpeechSynthesizer
from dashscope.legacy_1x.api_entities.aio_session import (
    close_shared_aio_session,
)
from dashscope.legacy_1x.api_entities.http_request import (
    close_shared_sync_session,
)
from dashscope.legacy_1x.common.env import (
    MAAS_REGIONS,
    api_key,
    api_key_file_path,
    base_compatible_api_url,
    base_http_api_url,
    base_websocket_api_url,
    validate_workspace_id,
)
from dashscope.legacy_1x.finetune.deployments import Deployments
from dashscope.legacy_1x.finetune.finetunes import FineTunes
from dashscope.legacy_1x.embeddings.batch_text_embedding import (
    BatchTextEmbedding,
)
from dashscope.legacy_1x.embeddings.batch_text_embedding_response import (
    BatchTextEmbeddingResponse,
)
from dashscope.legacy_1x.embeddings.multimodal_embedding import (
    MultiModalEmbedding,
    MultiModalEmbeddingItemAudio,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
    AioMultiModalEmbedding,
)
from dashscope.legacy_1x.embeddings.text_embedding import TextEmbedding
from dashscope.legacy_1x.files import Files
from dashscope.legacy_1x.models import Models
from dashscope.legacy_1x.nlp.understanding import Understanding
from dashscope.legacy_1x.rerank import AioTextReRank, TextReRank
from dashscope.legacy_1x.tokenizers import (
    Tokenization,
    Tokenizer,
    get_tokenizer,
    list_tokenizers,
)

# dashscope 2.0 public surface.
from dashscope import errors
from dashscope._config import Config, Settings, save_api_key, settings
from dashscope._quickstart import chat
from dashscope.client import AsyncClient, Client
from dashscope.errors import DashScopeError, ErrorCode
from dashscope.transport.ws_connection import WsConnection, WsTask

# cn-beijing defaults to the legacy dashscope.aliyuncs.com endpoints;
# its MaaS URLs are only reachable via an explicit set_region() call.
_MAAS_REGIONS = {*MAAS_REGIONS, "cn-beijing"}


def set_region(region: str, workspace_id: str = None):
    """Switch to a specific MaaS region.

    Updates base_http_api_url, base_compatible_api_url and
    base_websocket_api_url to point to the MaaS endpoint for the
    given region and workspace.

    Args:
        region (str): The MaaS region, e.g. "ap-southeast-1",
            "us-east-1", "cn-hongkong", "cn-beijing", "eu-central-1",
            "ap-northeast-1".
        workspace_id (str): The workspace ID, used as the subdomain
            of the MaaS endpoint.

    Raises:
        ValueError: If region is not supported, workspace_id is
            empty, or workspace_id is not a valid hostname label.
    """
    if region not in _MAAS_REGIONS:
        raise ValueError(
            f"Unsupported region '{region}'. "
            f"Supported regions: {sorted(_MAAS_REGIONS)}",
        )

    if not workspace_id:
        raise ValueError("workspace_id is required")

    validate_workspace_id(workspace_id)

    global base_http_api_url, base_compatible_api_url
    global base_websocket_api_url

    host = f"{workspace_id}.{region}.maas.aliyuncs.com"
    base_http_api_url = f"https://{host}/api/v1"
    base_compatible_api_url = f"https://{host}/compatible-mode/v1"
    base_websocket_api_url = f"wss://{host}/api-ws/v1/inference"


# The Assistants API (dashscope.legacy_1x.assistants /
# dashscope.legacy_1x.threads) is deprecated.
# Resolve its names lazily (PEP 562) so a plain `import dashscope` does not emit
# the DeprecationWarning for code that never touches them; accessing any of these
# names imports dashscope.legacy_1x.assistants, which warns once.
_DEPRECATED_LAZY_IMPORTS = {
    "Assistant": "dashscope.legacy_1x.assistants",
    "AssistantList": "dashscope.legacy_1x.assistants",
    "Assistants": "dashscope.legacy_1x.assistants",
    "AssistantFile": "dashscope.legacy_1x.assistants",
    "DeleteResponse": "dashscope.legacy_1x.assistants",
    "MessageFile": "dashscope.legacy_1x.threads",
    "Messages": "dashscope.legacy_1x.threads",
    "Run": "dashscope.legacy_1x.threads",
    "RunList": "dashscope.legacy_1x.threads",
    "Runs": "dashscope.legacy_1x.threads",
    "RunStep": "dashscope.legacy_1x.threads",
    "RunStepList": "dashscope.legacy_1x.threads",
    "Steps": "dashscope.legacy_1x.threads",
    "Thread": "dashscope.legacy_1x.threads",
    "ThreadMessage": "dashscope.legacy_1x.threads",
    "ThreadMessageList": "dashscope.legacy_1x.threads",
    "Threads": "dashscope.legacy_1x.threads",
}

_LEGACY_LAZY_MODULES = frozenset(
    {
        "aigc",
        "agentstudio",
        "api_entities",
        "app",
        "applications",
        "assistants",
        "audio",
        "cli",
        "common",
        "embeddings",
        "files",
        "finetune",
        "io",
        "model",
        "models",
        "multimodal",
        "nlp",
        "protocol",
        "rerank",
        "threads",
        "tokenizers",
        "utils",
    },
)


def __getattr__(name):
    module_name = _DEPRECATED_LAZY_IMPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
    elif name in _LEGACY_LAZY_MODULES:
        value = importlib.import_module(f"dashscope.{name}")
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}",
        )
    globals()[name] = value
    return value


def __dir__():
    return sorted(
        set(globals()) | set(_DEPRECATED_LAZY_IMPORTS) | _LEGACY_LAZY_MODULES,
    )


__all__ = [
    "__version__",
    "base_compatible_api_url",
    "base_http_api_url",
    "base_websocket_api_url",
    "api_key",
    "api_key_file_path",
    "save_api_key",
    "settings",
    "close_shared_aio_session",
    "close_shared_sync_session",
    "AioGeneration",
    "Conversation",
    "Generation",
    "History",
    "HistoryItem",
    "ImageSynthesis",
    "Transcription",
    "Files",
    "Deployments",
    "FineTunes",
    "Models",
    "TextEmbedding",
    "MultiModalEmbedding",
    "AioMultiModalEmbedding",
    "MultiModalEmbeddingItemAudio",
    "MultiModalEmbeddingItemImage",
    "MultiModalEmbeddingItemText",
    "SpeechSynthesizer",
    "HttpSpeechSynthesizer",
    "MultiModalConversation",
    "AioMultiModalConversation",
    "BatchTextEmbedding",
    "BatchTextEmbeddingResponse",
    "Understanding",
    "CodeGeneration",
    "Tokenization",
    "Tokenizer",
    "get_tokenizer",
    "list_tokenizers",
    "Application",
    "TextReRank",
    "AioTextReRank",
    "Assistants",
    "Threads",
    "Messages",
    "Runs",
    "Assistant",
    "ThreadMessage",
    "Run",
    "Steps",
    "AssistantList",
    "ThreadMessageList",
    "RunList",
    "RunStepList",
    "Thread",
    "DeleteResponse",
    "RunStep",
    "MessageFile",
    "AssistantFile",
    "VideoSynthesis",
    # dashscope 2.0
    "AsyncClient",
    "Client",
    "Config",
    "DashScopeError",
    "ErrorCode",
    "Settings",
    "WsConnection",
    "WsTask",
    "chat",
    "errors",
    "set_region",
]

logging.getLogger(__name__).addHandler(NullHandler())
