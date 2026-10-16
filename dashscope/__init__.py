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

from dashscope.version import __version__
from dashscope.aigc.code_generation import CodeGeneration
from dashscope.aigc.conversation import Conversation, History, HistoryItem
from dashscope.aigc.generation import AioGeneration, Generation
from dashscope.aigc.image_synthesis import ImageSynthesis
from dashscope.aigc.multimodal_conversation import (
    MultiModalConversation,
    AioMultiModalConversation,
)
from dashscope.aigc.video_synthesis import VideoSynthesis
from dashscope.app.application import Application
from dashscope.audio.asr.transcription import Transcription
from dashscope.audio.http_tts.http_speech_synthesizer import (
    HttpSpeechSynthesizer,
)
from dashscope.audio.tts.speech_synthesizer import SpeechSynthesizer
from dashscope.api_entities.aio_session import close_shared_aio_session
from dashscope.api_entities.http_request import close_shared_sync_session
from dashscope.common.api_key import save_api_key
from dashscope.common.env import (
    MAAS_REGIONS,
    api_key,
    api_key_file_path,
    base_compatible_api_url,
    base_http_api_url,
    base_websocket_api_url,
    validate_workspace_id,
)
from dashscope.finetune.deployments import Deployments
from dashscope.finetune.finetunes import FineTunes
from dashscope.embeddings.batch_text_embedding import BatchTextEmbedding
from dashscope.embeddings.batch_text_embedding_response import (
    BatchTextEmbeddingResponse,
)
from dashscope.embeddings.multimodal_embedding import (
    MultiModalEmbedding,
    MultiModalEmbeddingItemAudio,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
    AioMultiModalEmbedding,
)
from dashscope.embeddings.text_embedding import TextEmbedding
from dashscope.files import Files
from dashscope.models import Models
from dashscope.nlp.understanding import Understanding
from dashscope.rerank import AioTextReRank, TextReRank
from dashscope.tokenizers import (
    Tokenization,
    Tokenizer,
    get_tokenizer,
    list_tokenizers,
)

# dashscope 2.0 public surface (implementation lives in dashscope/v2/).
from dashscope.v2 import AsyncClient, Client, WsConnection, chat, settings

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


# The Assistants API (dashscope.assistants / dashscope.threads) is deprecated.
# Resolve its names lazily (PEP 562) so a plain `import dashscope` does not emit
# the DeprecationWarning for code that never touches them; accessing any of these
# names imports dashscope.assistants, which warns once.
_DEPRECATED_LAZY_IMPORTS = {
    "Assistant": "dashscope.assistants",
    "AssistantList": "dashscope.assistants",
    "Assistants": "dashscope.assistants",
    "AssistantFile": "dashscope.assistants",
    "DeleteResponse": "dashscope.assistants",
    "MessageFile": "dashscope.threads",
    "Messages": "dashscope.threads",
    "Run": "dashscope.threads",
    "RunList": "dashscope.threads",
    "Runs": "dashscope.threads",
    "RunStep": "dashscope.threads",
    "RunStepList": "dashscope.threads",
    "Steps": "dashscope.threads",
    "Thread": "dashscope.threads",
    "ThreadMessage": "dashscope.threads",
    "ThreadMessageList": "dashscope.threads",
    "Threads": "dashscope.threads",
}


def __getattr__(name):
    module_name = _DEPRECATED_LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}",
        )
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_DEPRECATED_LAZY_IMPORTS))


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
    "WsConnection",
    "chat",
    "set_region",
]

logging.getLogger(__name__).addHandler(NullHandler())
