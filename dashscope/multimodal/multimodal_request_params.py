# -*- coding: utf-8 -*-
# mypy: disable-error-code="annotation-unchecked"
from dataclasses import dataclass, field
import uuid


def get_random_uuid() -> str:
    """Generate and return a 32-character UUID string"""
    return uuid.uuid4().hex


@dataclass
class DashHeader:
    action: str
    task_id: str = field(default=get_random_uuid())
    streaming: str = field(default="duplex")  # default to duplex

    def to_dict(self):
        return {
            "action": self.action,
            "task_id": self.task_id,
            "request_id": self.task_id,
            "streaming": self.streaming,
        }


class DashPayloadParameters:
    def to_dict(self):
        pass


class DashPayloadInput:
    def to_dict(self):
        pass


@dataclass
class DashPayload:
    task_group: str = field(default="aigc")
    function: str = field(default="generation")
    model: str = field(default="")
    task: str = field(default="multimodal-generation")
    parameters: DashPayloadParameters = field(default=None)
    input: DashPayloadInput = field(default=None)

    def to_dict(self):
        payload = {
            "task_group": self.task_group,
            "function": self.function,
            "model": self.model,
            "task": self.task,
        }

        if self.parameters is not None:
            # pylint: disable=assignment-from-no-return
            payload["parameters"] = self.parameters.to_dict()

        if self.input is not None:
            # pylint: disable=assignment-from-no-return
            payload["input"] = self.input.to_dict()

        return payload


@dataclass
class RequestBodyInput(DashPayloadInput):
    workspace_id: str
    app_id: str
    directive: str
    dialog_id: str

    def to_dict(self):
        return {
            "workspace_id": self.workspace_id,
            "app_id": self.app_id,
            "directive": self.directive,
            "dialog_id": self.dialog_id,
        }


@dataclass
class AsrPostProcessing:
    replace_words: list = field(default=None)  # type: ignore[arg-type]

    def to_dict(self):
        if self.replace_words is None:
            return None
        if len(self.replace_words) == 0:
            return None
        return {
            "replace_words": [word.to_dict() for word in self.replace_words],
        }


@dataclass
class ReplaceWord:
    source: str = field(default=None)
    target: str = field(default=None)
    match_mode: str = field(default=None)

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "match_mode": self.match_mode,
        }


@dataclass
class Upstream:
    """struct for upstream"""

    audio_format: str = field(
        default="pcm",
    )  # upstream audio format, default pcm, supports pcm/opus
    type: str = field(
        default="AudioOnly",
    )  # upstream type: AudioOnly for voice only;
    # AudioAndVideo for video upload
    mode: str = field(
        default="tap2talk",
    )  # client interaction mode: push2talk/tap2talk/duplex
    sample_rate: int = field(default=16000)  # audio sample rate
    vocabulary_id: str = field(default=None)
    asr_post_processing: AsrPostProcessing = field(default=None)
    pass_through_params: dict = field(default=None)  # type: ignore[arg-type]

    def to_dict(self):
        upstream: dict = {
            "type": self.type,
            "mode": self.mode,
            "audio_format": self.audio_format,
            "sample_rate": self.sample_rate,
            "vocabulary_id": self.vocabulary_id,
        }
        if self.asr_post_processing is not None:
            upstream[
                "asr_post_processing"
            ] = self.asr_post_processing.to_dict()

        if self.pass_through_params is not None:
            upstream.update(self.pass_through_params)
        return upstream


@dataclass
class Downstream:
    # transcript  returns user speech recognition results
    # dialog  returns dialog system intermediate results
    # Multiple values can be set, comma-separated, default is transcript
    voice: str = field(default="")  # voice timbre
    sample_rate: int = field(
        default=0,
    )  # voice timbre # synthesis audio sample rate
    intermediate_text: str = field(
        default="transcript",
    )  # Controls which intermediate text is returned to user:
    debug: bool = field(default=False)  # Controls whether to return debug info
    # type_: str = field(default="Audio", metadata={"alias": "type"})  # downstream type: Text: no audio output; Audio: output audio, default  # noqa: E501  # pylint: disable=line-too-long
    audio_format: str = field(
        default="pcm",
    )  # downstream audio format, default pcm, supports pcm/mp3
    volume: int = field(default=50)  # voice volume 0-100
    pitch_rate: int = field(default=100)  # voice pitch 50-200
    speech_rate: int = field(default=100)  # voice speed 50-200
    pass_through_params: dict = field(default=None)  # type: ignore[arg-type]

    def to_dict(self):
        stream: dict = {
            "intermediate_text": self.intermediate_text,
            "debug": self.debug,
            # "type": self.type_,
            "audio_format": self.audio_format,
            "volume": self.volume,
            "pitch_rate": self.pitch_rate,
            "speech_rate": self.speech_rate,
        }
        if self.voice != "":
            stream["voice"] = self.voice
        if self.sample_rate != 0:
            stream["sample_rate"] = self.sample_rate
        if self.pass_through_params is not None:
            stream.update(self.pass_through_params)
        return stream


@dataclass
class DialogAttributes:
    agent_id: str = field(default=None)
    prompt: str = field(default=None)
    vocabulary_id: str = field(default=None)

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "prompt": self.prompt,
            "vocabulary_id": self.vocabulary_id,
        }


@dataclass
class Locations:
    city_name: str = field(default=None)
    latitude: str = field(default=None)
    longitude: str = field(default=None)

    def to_dict(self):
        return {
            "city_name": self.city_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


@dataclass
class Network:
    ip: str = field(default=None)

    def to_dict(self):
        return {
            "ip": self.ip,
        }


@dataclass
class Device:
    uuid: str = field(default=None)

    def to_dict(self):
        return {
            "uuid": self.uuid,
        }


@dataclass
class ClientInfo:
    user_id: str
    device: Device = field(default=None)
    network: Network = field(default=None)
    location: Locations = field(default=None)

    def to_dict(self):
        info = {
            "user_id": self.user_id,
            "sdk": "python",
        }
        if self.device is not None:
            info["device"] = self.device.to_dict()
        if self.network is not None:
            info["network"] = self.network.to_dict()
        if self.location is not None:
            info["location"] = self.location.to_dict()
        return info


@dataclass
class BizParams:
    user_defined_params: dict = field(default=None)  # type: ignore[arg-type]
    user_defined_tokens: dict = field(default=None)  # type: ignore[arg-type]
    tool_prompts: dict = field(default=None)  # type: ignore[arg-type]
    user_prompt_params: dict = field(default=None)  # type: ignore[arg-type]
    user_query_params: dict = field(default=None)  # type: ignore[arg-type]
    videos: list = field(default=None)  # type: ignore[arg-type]
    pass_through_params: dict = field(default=None)  # type: ignore[arg-type]

    def to_dict(self):
        params = {}
        if self.user_defined_params is not None:
            params["user_defined_params"] = self.user_defined_params
        if self.user_defined_tokens is not None:
            params["user_defined_tokens"] = self.user_defined_tokens
        if self.tool_prompts is not None:
            params["tool_prompts"] = self.tool_prompts
        if self.user_prompt_params is not None:
            params["user_prompt_params"] = self.user_prompt_params
        if self.user_query_params is not None:
            params["user_query_params"] = self.user_query_params
        if self.videos is not None:
            params["videos"] = self.videos
        if self.pass_through_params is not None:
            params.update(self.pass_through_params)
        return params


@dataclass
class RequestParameters(DashPayloadParameters):
    upstream: Upstream
    downstream: Downstream
    client_info: ClientInfo
    dialog_attributes: DialogAttributes = field(default=None)
    biz_params: BizParams = field(default=None)

    def to_dict(self):
        params = {
            "upstream": self.upstream.to_dict(),
            "downstream": self.downstream.to_dict(),
            "client_info": self.client_info.to_dict(),
        }

        if self.dialog_attributes is not None:
            params["dialog_attributes"] = self.dialog_attributes.to_dict()
        if self.biz_params is not None:
            params["biz_params"] = self.biz_params.to_dict()
        return params


@dataclass
class RequestToRespondParameters(DashPayloadParameters):
    images: list = field(default=None)  # type: ignore[arg-type]
    biz_params: BizParams = field(default=None)

    def to_dict(self):
        params = {}
        if self.images is not None:
            params["images"] = self.images
        if self.biz_params is not None:
            params["biz_params"] = self.biz_params.to_dict()
        return params


@dataclass
class RequestToRespondBodyInput(DashPayloadInput):
    app_id: str
    directive: str
    dialog_id: str
    type_: str = field(metadata={"alias": "type"}, default=None)
    text: str = field(default="")

    def to_dict(self):
        return {
            "app_id": self.app_id,
            "directive": self.directive,
            "dialog_id": self.dialog_id,
            "type": self.type_,
            "text": self.text,
        }
