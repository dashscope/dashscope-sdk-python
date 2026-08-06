# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import threading
import time
from dataclasses import field, dataclass
from typing import List, Any, Dict, Union
import uuid
from enum import Enum, unique

import websocket  # pylint: disable=wrong-import-order

import dashscope
from dashscope.common.error import ModelRequired
from dashscope.common.logging import logger
from dashscope.common.utils import get_user_agent


class OmniRealtimeCallback:
    """
    An interface that defines callback methods for getting omni-realtime results. # noqa E501
    Derive from this class and implement its function to provide your own data.
    """

    def on_open(self) -> None:
        pass

    def on_close(self, close_status_code, close_msg) -> None:
        pass

    def on_event(self, message: str) -> None:
        pass


@dataclass
class TranslationParams:
    """
    TranslationParams
    """

    @dataclass
    class Corpus:
        phrases: Dict[str, Any] = field(default=None)  # type: ignore[arg-type]

    language: str = field(default=None)
    corpus: Corpus = field(default=None)


@dataclass
class TranscriptionParams:
    """
    TranscriptionParams
    """

    language: str = field(default=None)
    sample_rate: int = field(default=16000)
    input_audio_format: str = field(default="pcm")
    corpus: Dict[str, Any] = field(default=None)  # type: ignore[arg-type]
    corpus_text: str = field(default=None)


@unique
class AudioFormat(Enum):
    # format, sample_rate, channels, bit_rate, name
    PCM_16000HZ_MONO_16BIT = ("pcm", 16000, "mono", "16bit", "pcm16")
    PCM_24000HZ_MONO_16BIT = ("pcm", 24000, "mono", "16bit", "pcm16")

    def __init__(  # pylint: disable=redefined-builtin
        self,
        format,
        sample_rate,
        channels,
        bit_rate,
        format_str,
    ):
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_rate = bit_rate
        self.format_str = format_str

    def __repr__(self):
        return self.format_str

    def __str__(self):
        return f"{self.format.upper()} with {self.sample_rate}Hz sample rate, {self.channels} channel, {self.bit_rate} bit rate:  {self.format_str}"  # noqa: E501  # pylint: disable=line-too-long


@unique
class AudioFormatType(Enum):
    """
    Preset audio format types for omni realtime input/output audio.

    These are provided for convenient reference only. ``AudioFormatConfig``
    also accepts any raw string value (e.g. ``"mp3"``) without validation,
    so new formats can be used without upgrading the SDK.
    """

    PCM = "pcm"
    WAV = "wav"

    def __str__(self):
        return self.value


@unique
class AudioSampleRate(Enum):
    """
    Preset audio sample rates(Hz) for omni realtime input/output audio.

    These are provided for convenient reference only. ``AudioFormatConfig``
    also accepts any raw int value without validation.
    """

    SAMPLE_RATE_8K = 8000
    SAMPLE_RATE_16K = 16000
    SAMPLE_RATE_24K = 24000
    SAMPLE_RATE_48K = 48000

    def __int__(self):
        return self.value


@dataclass
class AudioFormatConfig:
    """
    Audio format config for input(uplink) / output(downlink) audio in a
    omni realtime session.

    No strict validation is performed: the preset enums are only for
    convenient reference, while any raw value is also accepted so that new
    formats or sample rates can be used without upgrading the SDK.

    Parameters
    ----------
    type: str
        audio format type. Presets ``AudioFormatType.PCM`` / ``.WAV`` are
        provided for convenience; accepts an ``AudioFormatType`` enum or any
        raw string such as ``"pcm"`` / ``"wav"`` / ``"mp3"``.
    sample_rate: int
        audio sample rate in Hz. Presets ``AudioSampleRate`` (8000 / 16000 /
        24000 / 48000) are provided for convenience; accepts an
        ``AudioSampleRate`` enum or any raw int value. Defaults to 16000,
        which suits input audio; for output audio the server-side default
        is typically 24000, set it explicitly if needed.
    extra_params: Dict[str, Any]
        free extension parameters that will be merged into the ``format``
        dict. Reserved for future parameters such as speech rate, e.g.
        ``{"speech_rate": 1.2}``. Merged last, so identical keys (including
        ``type`` / ``sample_rate``) override the dataclass fields.
    """

    type: Union[str, AudioFormatType] = AudioFormatType.PCM.value
    sample_rate: Union[
        int,
        AudioSampleRate,
    ] = AudioSampleRate.SAMPLE_RATE_16K.value
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        convert to the ``format`` dict used in the session.update request.

        Preset enums are unwrapped to their underlying value; any raw value
        is passed through as-is without validation.
        """
        format_type = self.type
        if isinstance(format_type, AudioFormatType):
            format_type = format_type.value
        sample_rate = self.sample_rate
        if isinstance(sample_rate, AudioSampleRate):
            sample_rate = sample_rate.value
        result: Dict[str, Any] = {
            "type": format_type,
            "sample_rate": sample_rate,
        }
        if self.extra_params:
            result.update(self.extra_params)
        return result


class MultiModality(Enum):
    """
    MultiModality
    """

    TEXT = "text"
    AUDIO = "audio"

    def __str__(self):
        return self.name


class OmniRealtimeConversation:
    def __init__(
        self,
        model,
        callback: OmniRealtimeCallback,
        headers=None,
        workspace=None,
        url=None,
        api_key: str = None,
        additional_params=None,  # pylint: disable=unused-argument
    ):
        """
        Qwen Omni Realtime SDK
        Parameters:
        -----------
        model: str
            Model name.
        headers: Dict
            User-defined headers.
        callback: OmniRealtimeCallback
            Callback to receive real-time omni results.
        workspace: str
            Dashscope workspace ID.
        url: str
            Dashscope WebSocket URL.
        additional_params: Dict
            Additional parameters for the Dashscope API.
        """

        if model is None:
            raise ModelRequired("Model is required!")
        if callback is None:
            raise ModelRequired("Callback is required!")
        if url is None:
            url = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model}"  # noqa: E501
        else:
            url = f"{url}?model={model}"
        self.url = url
        self.apikey = api_key or dashscope.api_key
        self.user_headers = headers
        self.user_workspace = workspace
        self.model = model
        self.config = {}
        self.callback = callback
        self.ws = None
        self.session_id = None
        self.last_message = None
        self.last_response_id = None
        self.last_response_create_time = None
        self.last_first_text_delay = None
        self.last_first_audio_delay = None
        self.metrics = []
        # Add event for synchronously waiting on connection close
        self.disconnect_event = None
        self._disconnect_error = None

    def _generate_event_id(self):
        """
        generate random event id: event_xxxx
        """
        return "event_" + uuid.uuid4().hex

    def _get_websocket_header(self):
        ua = get_user_agent()
        headers = {
            "user-agent": ua,
            "Authorization": "Bearer " + self.apikey,
        }
        if self.user_headers:
            headers = {**self.user_headers, **headers}
        if self.user_workspace:
            headers = {
                **headers,
                "X-DashScope-WorkSpace": self.user_workspace,
            }
        return headers

    def connect(self) -> None:
        """
        connect to server, create session and return default session configuration  # noqa: E501
        """
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self._get_websocket_header(),
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
        timeout = 5  # max wait time in seconds
        start_time = time.time()
        while (
            not (self.ws.sock and self.ws.sock.connected)
            and (time.time() - start_time) < timeout
        ):
            time.sleep(0.1)  # Brief sleep to avoid busy polling
        if not (self.ws.sock and self.ws.sock.connected):
            raise TimeoutError(
                "websocket connection could not established within 5s. "
                "Please check your network connection, firewall settings, "
                "or server status.",
            )
        self.callback.on_open()

    def __send_str(self, data: str, enable_log: bool = True):
        if enable_log:
            logger.debug("[omni realtime] send string: %s", data)
        # Check connection status before sending
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            raise ConnectionError(
                "WebSocket connection is not established or has been closed. "
                "Please call connect() first and ensure the connection is "
                "active.",
            )
        self.ws.send(data)

    def create_item(self, item: dict):
        """
        send item.create request
        """
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "conversation.item.create",
                    "item": item,
                },
            ),
            enable_log=True,
        )

    def _apply_audio_format(
        self,
        input_audio_format: AudioFormat,
        output_audio_format: AudioFormat,
        input_audio_config: AudioFormatConfig,
        output_audio_config: AudioFormatConfig,
    ) -> None:
        """
        apply audio format config into ``self.config``.

        Prefer the new-style ``session.audio.{input,output}.format`` structure
        when ``input_audio_config`` / ``output_audio_config`` is provided,
        otherwise keep the old-style top-level fields for backward
        compatibility.

        When only one side is provided, the other side is omitted from the
        request and the server-side default applies (input: pcm/16000,
        output: pcm/24000); the legacy top-level fields are NOT mixed in,
        to avoid sending both structures in one request.
        """
        if input_audio_config is None and output_audio_config is None:
            # old-style: keep backward compatibility
            self.config["input_audio_format"] = input_audio_format.format_str
            self.config["output_audio_format"] = output_audio_format.format_str
            return
        # new-style: separate audio format type and sample rate under
        # session.audio.{input,output}.format
        audio: Dict[str, Any] = {}
        if input_audio_config is not None:
            audio["input"] = {"format": input_audio_config.to_dict()}
        if output_audio_config is not None:
            audio["output"] = {"format": output_audio_config.to_dict()}
        self.config["audio"] = audio

    def _apply_transcription_params(
        self,
        transcription_params: TranscriptionParams,
    ) -> None:
        """
        apply transcription params into ``self.config``.

        The legacy top-level ``input_audio_format`` / ``sample_rate`` fields
        are skipped when the new-style ``audio`` structure is present, to
        avoid sending conflicting audio format fields in one request.
        """
        self.config["input_audio_transcription"] = {}
        if transcription_params.language is not None:
            self.config["input_audio_transcription"].update(
                {"language": transcription_params.language},
            )
        if transcription_params.corpus_text is not None:
            transcription_params.corpus = {
                "text": transcription_params.corpus_text,
            }
        if transcription_params.corpus is not None:
            self.config["input_audio_transcription"].update(
                {"corpus": transcription_params.corpus},
            )
        if "audio" in self.config:
            logger.warning(
                "[omni realtime] input_audio_config/output_audio_config is "
                "set, skip top-level input_audio_format/sample_rate from "
                "transcription_params to avoid conflicting audio formats",
            )
            return
        self.config[
            "input_audio_format"
        ] = transcription_params.input_audio_format
        self.config["sample_rate"] = transcription_params.sample_rate

    def update_session(
        self,
        output_modalities: List[MultiModality],
        voice: str = None,
        input_audio_format: AudioFormat = AudioFormat.PCM_16000HZ_MONO_16BIT,
        output_audio_format: AudioFormat = AudioFormat.PCM_24000HZ_MONO_16BIT,
        enable_input_audio_transcription: bool = True,
        input_audio_transcription_model: str = None,
        enable_turn_detection: bool = True,
        turn_detection_type: str = "server_vad",
        prefix_padding_ms: int = 300,
        turn_detection_threshold: float = 0.2,
        turn_detection_silence_duration_ms: int = 800,
        turn_detection_param: dict = None,
        translation_params: TranslationParams = None,
        transcription_params: TranscriptionParams = None,
        input_audio_config: AudioFormatConfig = None,
        output_audio_config: AudioFormatConfig = None,
        **kwargs,
    ) -> None:
        """
        update session configuration, should be used before create response

        Parameters
        ----------
        output_modalities: list[MultiModality]
            omni output modalities to be used in session
        voice: str
            voice to be used in session
        input_audio_format: AudioFormat
            input audio format. Legacy parameter, kept for backward
            compatibility and still the default path. Prefer
            ``input_audio_config`` which supports separate audio format
            type and sample rate.
        output_audio_format: AudioFormat
            output audio format. Legacy parameter, kept for backward
            compatibility and still the default path. Prefer
            ``output_audio_config`` which supports separate audio format
            type and sample rate.
        enable_turn_detection: bool
            enable turn detection
        turn_detection_threshold: float
            turn detection threshold, range [-1, 1]
            In a noisy environment, it may be necessary to increase the threshold to reduce false detections  # noqa: E501  # pylint: disable=line-too-long
            In a quiet environment, it may be necessary to decrease the threshold to improve sensitivity  # noqa: E501  # pylint: disable=line-too-long
        turn_detection_silence_duration_ms: int
            duration of silence in milliseconds to detect turn, range [200, 6000]  # noqa: E501
        translation_params: TranslationParams
            translation params, include language. Only effective with qwen3-livetranslate-flash-realtime model or  # noqa: E501  # pylint: disable=line-too-long
             further models. Do not set this parameter for other models.
        transcription_params: TranscriptionParams
            transcription params, include language, sample_rate, input_audio_format, corpus.  # noqa: E501  # pylint: disable=line-too-long
            Only effective with qwen3-asr-flash-realtime model or
            further models. Do not set this parameter for other models.
        input_audio_config: AudioFormatConfig
            input(uplink) audio format config. Supports separate format type
            (pcm/wav) and sample rate (8000/16000/24000/48000), as well as
            free extension parameters via ``extra_params``. When provided,
            the request emits the ``session.audio.input.format`` structure.
        output_audio_config: AudioFormatConfig
            output(downlink) audio format config. Supports separate format
            type (pcm/wav) and sample rate (8000/16000/24000/48000), as well
            as free extension parameters via ``extra_params``. When provided,
            the request emits the ``session.audio.output.format`` structure.

        Notes
        -----
        Once either ``input_audio_config`` or ``output_audio_config`` is
        provided, the legacy top-level ``input_audio_format`` /
        ``output_audio_format`` fields are not sent; an unspecified side
        falls back to the server-side default. The top-level format fields
        from ``transcription_params`` are also skipped in this case.
        """
        self.config = {
            "modalities": [m.value for m in output_modalities],
            "voice": voice,
        }
        self._apply_audio_format(
            input_audio_format,
            output_audio_format,
            input_audio_config,
            output_audio_config,
        )
        if enable_input_audio_transcription:
            self.config["input_audio_transcription"] = {
                "model": input_audio_transcription_model,
            }
        else:
            self.config["input_audio_transcription"] = None
        if enable_turn_detection:
            self.config["turn_detection"] = {
                "type": turn_detection_type,
                "threshold": turn_detection_threshold,
                "prefix_padding_ms": prefix_padding_ms,
                "silence_duration_ms": turn_detection_silence_duration_ms,
            }
            if turn_detection_param is not None:
                self.config["turn_detection"].update(turn_detection_param)
        else:
            self.config["turn_detection"] = None
        if translation_params is not None:
            self.config["translation"] = {
                "language": translation_params.language,
            }
            if translation_params.corpus is not None:
                if (
                    translation_params.corpus
                    and translation_params.corpus.phrases is not None
                ):
                    self.config["translation"]["corpus"] = {
                        "phrases": translation_params.corpus.phrases,
                    }
        if transcription_params is not None:
            self._apply_transcription_params(transcription_params)
        self.config.update(kwargs)
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "session.update",
                    "session": self.config,
                },
            ),
        )

    def end_session(self, timeout: int = 20) -> None:
        """
        end session

        Parameters:
        -----------
        timeout: int
            Timeout in seconds to wait for the session to end. Default is 20 seconds.  # noqa: E501
        """
        if self.disconnect_event is not None:
            # if the event is already set, do nothing
            return

        # create the event
        self.disconnect_event = threading.Event()
        self._disconnect_error = None

        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "session.finish",
                },
            ),
        )

        # wait for the event to be set
        finish_success = self.disconnect_event.wait(timeout)
        error = self._disconnect_error
        self.disconnect_event = None
        self._disconnect_error = None

        # if the server returned an error or timed out, close the connection
        if error is not None:
            self.close()
            raise RuntimeError(f"Session ended with error: {error}")
        if not finish_success:
            self.close()
            raise TimeoutError(
                f"Session end timeout after {timeout} seconds",
            )

    def end_session_async(self) -> None:
        """
        end session asynchronously. you need close the connection manually
        """
        # Send end session message
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "session.finish",
                },
            ),
        )

    def append_audio(self, audio_b64: str) -> None:
        """
        send audio in base64 format

        Parameters
        ----------
        audio_b64: str
            base64 audio string
        """
        logger.debug("[omni realtime] append audio: %s", len(audio_b64))
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                },
            ),
            False,
        )

    def append_video(self, video_b64: str) -> None:
        """
        send one image frame in video in base64 format

        Parameters
        ----------
        video_b64: str
            base64 image string
        """
        logger.debug("[omni realtime] append video: %s", len(video_b64))
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "input_image_buffer.append",
                    "image": video_b64,
                },
            ),
            False,
        )

    def commit(self) -> None:
        """
        Commit the audio and video sent before.
        When in Server VAD mode, the client does not need to use this method,
        the server will commit the audio automatically after detecting vad end.
        """
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "input_audio_buffer.commit",
                },
            ),
        )

    def clear_appended_audio(self) -> None:
        """
        clear the audio sent to server before.
        """
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "input_audio_buffer.clear",
                },
            ),
        )

    def create_response(
        self,
        instructions: str = None,
        output_modalities: List[MultiModality] = None,
    ) -> None:
        """
        create response, use audio and video commited before to request llm.
        When in Server VAD mode, the client does not need to use this method,
        the server will create response automatically after detecting vad
        and sending commit.

        Parameters
        ----------
        instructions: str
            instructions to llm
        output_modalities: list[MultiModality]
            omni output modalities to be used in session
        """
        request = {
            "event_id": self._generate_event_id(),
            "type": "response.create",
            "response": {},
        }
        request["response"]["instructions"] = instructions
        if output_modalities:
            request["response"]["modalities"] = [
                m.value for m in output_modalities
            ]
        self.__send_str(json.dumps(request))

    def cancel_response(self) -> None:
        """
        cancel the current response
        """
        self.__send_str(
            json.dumps(
                {
                    "event_id": self._generate_event_id(),
                    "type": "response.cancel",
                },
            ),
        )

    def send_raw(self, raw_data: str) -> None:
        """
        send raw data to server
        """
        self.__send_str(raw_data)

    def close(self) -> None:
        """
        close the connection to server
        """
        self.ws.close()

    # Callback for listening to messages
    def _on_message(  # pylint: disable=unused-argument,too-many-branches
        self,
        ws,
        message,
    ):
        if isinstance(message, str):
            logger.debug(
                "[omni realtime] receive string %s",
                message[:1024],
            )
            try:
                # Attempt to parse message as JSON
                json_data = json.loads(message)
                self.last_message = json_data
                self.callback.on_event(json_data)
                if "type" in message:
                    if "session.created" == json_data["type"]:
                        logger.info("[omni realtime] session created")
                        self.session_id = json_data["session"]["id"]
                    elif "session.finished" == json_data["type"]:
                        # wait for the event to be set
                        logger.info("[omni realtime] session finished")
                        if self.disconnect_event is not None:
                            self.disconnect_event.set()
                    elif "error" == json_data.get("type"):
                        if self.disconnect_event is not None:
                            self._disconnect_error = json_data.get("error")
                            logger.warning(
                                "[omni realtime] error during end_session: %s",
                                self._disconnect_error,
                            )
                            self.disconnect_event.set()
                    if "response.created" == json_data["type"]:
                        self.last_response_id = json_data["response"]["id"]
                        self.last_response_create_time = time.time() * 1000
                        self.last_first_audio_delay = None
                        self.last_first_text_delay = None
                    elif (
                        "response.audio_transcript.delta" == json_data["type"]
                    ):
                        if (
                            self.last_response_create_time
                            and self.last_first_text_delay is None
                        ):
                            self.last_first_text_delay = (
                                time.time() * 1000
                                - self.last_response_create_time
                            )
                    elif "response.audio.delta" == json_data["type"]:
                        if (
                            self.last_response_create_time
                            and self.last_first_audio_delay is None
                        ):
                            self.last_first_audio_delay = (
                                time.time() * 1000
                                - self.last_response_create_time
                            )
                    elif "response.done" == json_data["type"]:
                        # pylint: disable=line-too-long
                        logger.info(
                            "[Metric] response: %s, first text delay: %s, first audio delay: %s",  # noqa: E501
                            self.last_response_id,
                            self.last_first_text_delay,
                            self.last_first_audio_delay,
                        )
            except json.JSONDecodeError:
                logger.error("Failed to parse message as JSON.")
                # pylint: disable=broad-exception-raised,raise-missing-from
                raise Exception("Failed to parse message as JSON.")
        elif isinstance(message, (bytes, bytearray)):
            # If parsing fails, treat as binary message
            logger.error(
                "should not receive binary message in omni realtime api",
            )
            logger.debug(
                "[omni realtime] receive binary %s bytes",
                len(message),
            )

    def _on_close(  # pylint: disable=unused-argument
        self,
        ws,
        close_status_code,
        close_msg,
    ):
        self.callback.on_close(close_status_code, close_msg)

    # Callback for WebSocket error
    def _on_error(self, ws, error):  # pylint: disable=unused-argument
        logger.error("websocket error: %s", error)
        # Do not raise exception here, let the connection close naturally
        # Raising exception in callback can cause unexpected thread termination

    # Get the taskId of the last task
    def get_session_id(self) -> str:
        return self.session_id  # type: ignore[return-value]

    def get_last_message(self) -> str:
        return self.last_message  # type: ignore[return-value]

    def get_last_response_id(self) -> str:
        return self.last_response_id  # type: ignore[return-value]

    def get_last_first_text_delay(self):
        return self.last_first_text_delay

    def get_last_first_audio_delay(self):
        return self.last_first_audio_delay
