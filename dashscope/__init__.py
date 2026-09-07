# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# pylint: disable=wrong-import-position

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
    api_key,
    api_key_file_path,
    base_compatible_api_url,
    base_http_api_url,
    base_websocket_api_url,
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
from dashscope.assistants import Assistant, AssistantList, Assistants
from dashscope.assistants.assistant_types import AssistantFile, DeleteResponse
from dashscope.threads import (
    MessageFile,
    Messages,
    Run,
    RunList,
    Runs,
    RunStep,
    RunStepList,
    Steps,
    Thread,
    ThreadMessage,
    ThreadMessageList,
    Threads,
)
from dashscope.tokenizers import (
    Tokenization,
    Tokenizer,
    get_tokenizer,
    list_tokenizers,
)


# Supported MaaS regions and their default base URLs
_MAAS_REGIONS = {
    "cn-beijing",
    "ap-southeast-1",
    "us-east-1",
    "cn-hongkong",
    "eu-central-1",
    "ap-northeast-1",
}


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
        ValueError: If region is not supported or workspace_id is
            empty.
    """
    if region not in _MAAS_REGIONS:
        raise ValueError(
            f"Unsupported region '{region}'. "
            f"Supported regions: {sorted(_MAAS_REGIONS)}",
        )

    if not workspace_id:
        raise ValueError("workspace_id is required")

    global base_http_api_url, base_compatible_api_url
    global base_websocket_api_url

    host = f"{workspace_id}.{region}.maas.aliyuncs.com"
    base_http_api_url = f"https://{host}/api/v1"
    base_compatible_api_url = f"https://{host}/compatible-mode/v1"
    base_websocket_api_url = f"wss://{host}/api-ws/v1/inference"


__all__ = [
    "__version__",
    "base_compatible_api_url",
    "base_http_api_url",
    "base_websocket_api_url",
    "api_key",
    "api_key_file_path",
    "save_api_key",
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
    "set_region",
]

logging.getLogger(__name__).addHandler(NullHandler())
