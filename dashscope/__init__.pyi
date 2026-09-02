# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Stub file for PEP 561 type information.

from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
api_key: str
api_key_file_path: str
base_compatible_api_url: str
base_http_api_url: str
base_websocket_api_url: str

def save_api_key(api_key: str, file_path: Optional[str] = ...) -> None: ...
def close_shared_aio_session() -> None: ...

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------
class Message:
    role: str
    content: Union[str, List[Any]]
    def __init__(
        self, role: str, content: Union[str, List[Any]] = None, **kwargs: Any
    ) -> None: ...

class DashScopeAPIResponse:
    status_code: int
    request_id: str
    code: str
    message: str
    output: Any
    usage: Any
    headers: Dict[str, str]
    def __init__(
        self,
        status_code: int,
        request_id: str = "",
        code: str = "",
        message: str = "",
        output: Any = None,
        usage: Any = None,
        headers: Dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None: ...

class GenerationResponse(DashScopeAPIResponse): ...
class MultiModalConversationResponse(DashScopeAPIResponse): ...
class ImageSynthesisResponse(DashScopeAPIResponse): ...
class VideoSynthesisResponse(DashScopeAPIResponse): ...
class ReRankResponse(DashScopeAPIResponse): ...
class TranscriptionResponse(DashScopeAPIResponse): ...
class ConversationResponse(DashScopeAPIResponse): ...
class BatchTextEmbeddingResponse: ...

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class ApplicationResponse: ...

# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------
class ResultCallback: ...
class SpeechSynthesisResult: ...

# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------
class Generation:
    @classmethod
    def call(
        cls,
        model: str,
        prompt: Any = None,
        history: list | None = None,
        api_key: str | None = None,
        messages: List[Message] | None = None,
        plugins: Union[str, Dict[str, Any], None] = None,
        workspace: str | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: Union[str, List[str], None] = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        result_format: str | None = None,
        incremental_output: bool | None = None,
        enable_search: bool | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: Union[str, Dict[str, Any], None] = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        n: int | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        search_options: Dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: Dict[str, Any] | None = None,
        output_format: str | None = None,
        **kwargs: Any,
    ) -> Union[
        GenerationResponse,
        Generator[GenerationResponse, None, None],
    ]: ...

class AioGeneration:
    @classmethod
    async def call(
        cls,
        model: str,
        prompt: Any = None,
        history: list | None = None,
        api_key: str | None = None,
        messages: List[Message] | None = None,
        **kwargs: Any,
    ) -> Union[
        GenerationResponse,
        Generator[GenerationResponse, None, None],
    ]: ...

# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
class HistoryItem: ...
class History: ...

class Conversation:
    def __init__(self, history: History | None = None) -> None: ...
    def call(
        self,
        model: str,
        prompt: Any,
        history: History | None = None,
        auto_history: bool = ...,
        n_history: int = ...,
        **kwargs: Any,
    ) -> Union[
        ConversationResponse,
        Generator[ConversationResponse, None, None],
    ]: ...

# ---------------------------------------------------------------------------
# MultiModal Conversation
# ---------------------------------------------------------------------------
class MultiModalConversation:
    @classmethod
    def call(
        cls,
        model: str,
        messages: List[Any] | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        text: str | None = None,
        voice: str | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: Union[str, List[str], None] = None,
        result_format: str | None = None,
        incremental_output: bool | None = None,
        enable_search: bool | None = None,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: Union[str, Dict[str, Any], None] = None,
        enable_thinking: bool | None = None,
        n: int | None = None,
        ocr_options: Dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        **kwargs: Any,
    ) -> Union[
        MultiModalConversationResponse,
        Generator[MultiModalConversationResponse, None, None],
    ]: ...

class AioMultiModalConversation:
    @classmethod
    async def call(
        cls,
        model: str,
        messages: List[Any] | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Union[
        MultiModalConversationResponse,
        Generator[MultiModalConversationResponse, None, None],
    ]: ...

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class TextEmbedding:
    @classmethod
    def call(
        cls,
        model: str,
        input: Union[str, List[str]],
        workspace: str | None = None,
        api_key: str | None = None,
        text_type: str | None = None,
        dimension: int | None = None,
        output_type: str | None = None,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> DashScopeAPIResponse: ...

class MultiModalEmbeddingItemText: ...
class MultiModalEmbeddingItemImage: ...
class MultiModalEmbeddingItemAudio: ...

class MultiModalEmbedding:
    @classmethod
    def call(
        cls,
        model: str,
        input: Any,
        api_key: str | None = None,
        workspace: str | None = None,
        dimension: int | None = None,
        **kwargs: Any,
    ) -> DashScopeAPIResponse: ...

class AioMultiModalEmbedding:
    @classmethod
    async def call(
        cls,
        model: str,
        input: Any,
        api_key: str | None = None,
        workspace: str | None = None,
        dimension: int | None = None,
        **kwargs: Any,
    ) -> DashScopeAPIResponse: ...

class BatchTextEmbedding:
    @classmethod
    def call(
        cls,
        model: str,
        url: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> BatchTextEmbeddingResponse: ...
    @classmethod
    def async_call(
        cls,
        model: str,
        url: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> BatchTextEmbeddingResponse: ...
    @classmethod
    def fetch(
        cls,
        task: Any,
        api_key: str | None = None,
        workspace: str | None = None,
    ) -> BatchTextEmbeddingResponse: ...
    @classmethod
    def wait(
        cls,
        task: Any,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> BatchTextEmbeddingResponse: ...
    @classmethod
    def cancel(
        cls,
        task: Any,
        api_key: str | None = None,
        workspace: str | None = None,
    ) -> DashScopeAPIResponse: ...

# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------
class TextReRank:
    @classmethod
    def call(
        cls,
        model: str,
        query: str,
        documents: List[str],
        return_documents: bool | None = None,
        top_n: int | None = None,
        api_key: str | None = None,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> ReRankResponse: ...

class AioTextReRank:
    @classmethod
    async def call(
        cls,
        model: str,
        query: str,
        documents: List[str],
        return_documents: bool | None = None,
        top_n: int | None = None,
        api_key: str | None = None,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> ReRankResponse: ...

# ---------------------------------------------------------------------------
# Image Synthesis
# ---------------------------------------------------------------------------
class ImageSynthesis:
    @classmethod
    def call(
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] | None = None,
        api_key: str | None = None,
        size: str | None = None,
        n: int | None = None,
        seed: int | None = None,
        style: str | None = None,
        ref_strength: float | None = None,
        prompt_extend: bool | None = None,
        watermark: bool | None = None,
        **kwargs: Any,
    ) -> ImageSynthesisResponse: ...
    @classmethod
    def async_call(
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> ImageSynthesisResponse: ...
    @classmethod
    def sync_call(
        cls,
        model: str,
        prompt: Any,
        negative_prompt: Any = None,
        images: List[str] | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> ImageSynthesisResponse: ...
    @classmethod
    def fetch(
        cls,
        task: Union[str, ImageSynthesisResponse],
        api_key: str | None = None,
        workspace: str | None = None,
    ) -> ImageSynthesisResponse: ...
    @classmethod
    def wait(
        cls,
        task: Union[str, ImageSynthesisResponse],
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> ImageSynthesisResponse: ...

# ---------------------------------------------------------------------------
# Video Synthesis
# ---------------------------------------------------------------------------
class VideoSynthesis:
    @classmethod
    def call(
        cls,
        model: str,
        prompt: Any = None,
        extend_prompt: bool = True,
        negative_prompt: str | None = None,
        img_url: str | None = None,
        audio_url: str | None = None,
        reference_video_urls: List[str] | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        size: str | None = None,
        duration: int | None = None,
        seed: int | None = None,
        prompt_extend: bool | None = None,
        watermark: bool | None = None,
        resolution: str | None = None,
        ratio: str | None = None,
        **kwargs: Any,
    ) -> VideoSynthesisResponse: ...
    @classmethod
    def async_call(
        cls,
        model: str,
        prompt: Any = None,
        img_url: str | None = None,
        audio_url: str | None = None,
        reference_video_urls: List[str] | None = None,
        extend_prompt: bool = True,
        negative_prompt: str | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        size: str | None = None,
        duration: int | None = None,
        seed: int | None = None,
        prompt_extend: bool | None = None,
        watermark: bool | None = None,
        resolution: str | None = None,
        ratio: str | None = None,
        **kwargs: Any,
    ) -> VideoSynthesisResponse: ...
    @classmethod
    def fetch(
        cls,
        task: Union[str, VideoSynthesisResponse],
        api_key: str | None = None,
        workspace: str | None = None,
    ) -> VideoSynthesisResponse: ...
    @classmethod
    def wait(
        cls,
        task: Union[str, VideoSynthesisResponse],
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> VideoSynthesisResponse: ...
    @classmethod
    def cancel(
        cls,
        task: Union[str, VideoSynthesisResponse],
        api_key: str | None = None,
        workspace: str | None = None,
    ) -> DashScopeAPIResponse: ...

# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------
class SpeechSynthesizer:
    @classmethod
    def call(
        cls,
        model: str,
        text: str,
        callback: ResultCallback | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> SpeechSynthesisResult: ...

class HttpSpeechSynthesizer: ...

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
class Transcription:
    @classmethod
    def call(
        cls,
        model: str,
        file_urls: List[str],
        phrase_id: str | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse: ...
    @classmethod
    def async_call(
        cls,
        model: str,
        file_urls: List[str],
        phrase_id: str | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse: ...
    @classmethod
    def fetch(
        cls,
        task: Any,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse: ...
    @classmethod
    def wait(
        cls,
        task: Any,
        api_key: str | None = None,
        workspace: str | None = None,
        wait_timeout: int | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse: ...

# ---------------------------------------------------------------------------
# Fine-tuning & Deployments
# ---------------------------------------------------------------------------
class FineTunes:
    @classmethod
    def call(
        cls,
        model: str,
        training_file_ids: Union[list, str],
        validation_file_ids: Union[list, str] | None = None,
        mode: str | None = None,
        hyper_parameters: dict | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def cancel(
        cls,
        job_id: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def list(
        cls,
        page_no: int = ...,
        page_size: int = ...,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def get(
        cls,
        job_id: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def delete(
        cls,
        job_id: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...

class Deployments:
    @classmethod
    def call(
        cls,
        model: str,
        capacity: int,
        version: str,
        suffix: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def list(
        cls,
        page_no: int = ...,
        page_size: int = ...,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def get(
        cls,
        deployed_model: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def delete(
        cls,
        deployed_model: str,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def scale(
        cls,
        deployed_model: str,
        capacity: int,
        api_key: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> Any: ...

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class Application:
    @classmethod
    def call(
        cls,
        app_id: str,
        prompt: str | None = None,
        history: list | None = None,
        workspace: str | None = None,
        api_key: str | None = None,
        messages: List[Message] | None = None,
        **kwargs: Any,
    ) -> Union[
        ApplicationResponse,
        Generator[ApplicationResponse, None, None],
    ]: ...

# ---------------------------------------------------------------------------
# Assistants
# ---------------------------------------------------------------------------
class Assistant: ...
class AssistantList: ...

class Assistants:
    @classmethod
    def create(
        cls,
        model: str,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Assistant: ...
    @classmethod
    def retrieve(
        cls,
        assistant_id: str,
        workspace: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Assistant: ...
    @classmethod
    def list(
        cls,
        limit: int | None = None,
        order: str | None = None,
        after: str | None = None,
        before: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> AssistantList: ...

# ---------------------------------------------------------------------------
# Threads (deprecated)
# ---------------------------------------------------------------------------
class Thread: ...
class Threads: ...
class ThreadMessage: ...
class ThreadMessageList: ...
class Messages: ...
class Run: ...
class RunList: ...
class Runs: ...
class RunStep: ...
class RunStepList: ...
class Steps: ...
class MessageFile: ...
class AssistantFile: ...
class DeleteResponse: ...

# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------
class CodeGeneration:
    @classmethod
    def call(cls, model: str, prompt: str, **kwargs: Any) -> Any: ...

class Understanding: ...
class Models: ...
class Files: ...

# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class Tokenization:
    @classmethod
    def call(
        cls, model: str, input: Any, **kwargs: Any
    ) -> DashScopeAPIResponse: ...

class Tokenizer:
    def encode(self, text: str, **kwargs: Any) -> List[int]: ...
    def decode(self, token_ids: List[int], **kwargs: Any) -> str: ...

def get_tokenizer(model: str, **kwargs: Any) -> Tokenizer: ...
def list_tokenizers() -> List[str]: ...

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__: list[str]
