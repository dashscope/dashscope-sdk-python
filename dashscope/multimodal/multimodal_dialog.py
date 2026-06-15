# -*- coding: utf-8 -*-
import json
import time
import threading
from abc import abstractmethod

import websocket

import dashscope
from dashscope.common.logging import logger
from dashscope.common.error import InputRequired
from dashscope.common.utils import get_user_agent
from dashscope.multimodal import dialog_state
from dashscope.multimodal.multimodal_constants import (
    RESPONSE_NAME_STARTED,
    RESPONSE_NAME_STOPPED,
    RESPONSE_NAME_STATE_CHANGED,
    RESPONSE_NAME_REQUEST_ACCEPTED,
    RESPONSE_NAME_SPEECH_STARTED,
    RESPONSE_NAME_SPEECH_ENDED,
    RESPONSE_NAME_RESPONDING_STARTED,
    RESPONSE_NAME_RESPONDING_ENDED,
    RESPONSE_NAME_SPEECH_CONTENT,
    RESPONSE_NAME_RESPONDING_CONTENT,
    RESPONSE_NAME_ERROR,
    RESPONSE_NAME_HEART_BEAT,
)
from dashscope.multimodal.multimodal_request_params import (
    RequestParameters,
    get_random_uuid,
    DashHeader,
    RequestBodyInput,
    DashPayload,
    RequestToRespondParameters,
    RequestToRespondBodyInput,
)
from dashscope.protocol.websocket import ActionType


class MultiModalCallback:
    """
    Voice chat callback class for handling various events during voice chat.
    """

    def on_started(self, dialog_id: str) -> None:
        """
        Notify dialog started.

        :param dialog_id: Callback dialog ID
        """

    def on_stopped(self) -> None:
        """
        Notify dialog stopped.
        """

    def on_state_changed(self, state: "dialog_state.DialogState") -> None:
        """
        Dialog state changed.

        :param state: New dialog state
        """

    def on_speech_audio_data(self, data: bytes) -> None:
        """
        Synthesized audio data callback.

        :param data: Audio data
        """

    def on_error(self, error) -> None:
        """
        Called when an error occurs.

        :param error: Error message
        """

    def on_connected(self) -> None:
        """
        Called after successfully connecting to the server.
        """

    def on_responding_started(self):
        """
        Response started callback.
        """

    def on_responding_ended(self, payload):
        """
        Response ended.
        """

    def on_speech_started(self):
        """
        Speech input started.
        """

    def on_speech_ended(self):
        """
        Speech input ended.
        """

    def on_speech_content(self, payload):
        """
        Speech recognition text.

        :param payload: text
        """

    def on_responding_content(self, payload):
        """
        LLM response text.

        :param payload: text
        """

    def on_request_accepted(self):
        """
        Interrupt request accepted.
        """

    def on_close(self, close_status_code, close_msg):
        """
        Called when connection is closed.

        :param close_status_code: Close status code
        :param close_msg: Close message
        """


class MultiModalDialog:
    """
    Service class for managing WebSocket connections for voice chat.
    """

    def __init__(
        self,
        app_id: str,
        request_params: RequestParameters,
        multimodal_callback: MultiModalCallback,
        workspace_id: str = None,
        url: str = None,
        api_key: str = None,
        dialog_id: str = None,
        model: str = None,
    ):
        """
        Create a voice dialog session.

        This method initializes a new voice_chat session, setting up
        the necessary parameters to start interacting with the model.
        :param workspace_id: Customer workspace_id, primary workspace ID,
            optional field
        :param app_id: Application ID created in the console, used to
            determine which dialog system to use
        :param request_params: Request parameter collection
        :param url: (str) API URL address.
        :param multimodal_callback: (MultimodalCallback) Callback object
            for processing messages from server.
        :param api_key: (str) Application unique access key
        :param dialog_id: Dialog ID, if provided, continues the
            conversation with previous context
        :param model: Model
        """
        if request_params is None:
            raise InputRequired("request_params is required!")
        if url is None:
            url = dashscope.base_websocket_api_url
        if api_key is None:
            api_key = dashscope.api_key

        self.request_params = request_params
        self.model = model
        self._voice_detection = None
        self.thread = None
        self.ws = None
        self.request = _Request()
        self._callback = multimodal_callback
        self.url = url
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.app_id = app_id
        self.dialog_id = dialog_id
        self.dialog_state = dialog_state.StateMachine()
        self.response = _Response(
            self.dialog_state,
            self._callback,
            self.close,
        )  # pass self.close as callback

    def _on_message(  # pylint: disable=unused-argument
        self,
        ws,
        message,
    ):
        logger.debug(f"<<<<<<< Received message: {message}")
        if isinstance(message, str):
            self.response.handle_text_response(message)
        elif isinstance(message, (bytes, bytearray)):
            self.response.handle_binary_response(message)

    def _on_error(self, ws, error):  # pylint: disable=unused-argument
        logger.error(f"Error: {error}")
        if self._callback:
            self._callback.on_error(error)

    def _on_close(  # pylint: disable=unused-argument
        self,
        ws,
        close_status_code,
        close_msg,
    ):
        try:
            logger.debug(
                "WebSocket connection closed with status %s and message %s",  # noqa: E501
                close_status_code,
                close_msg,
            )
            if close_status_code is None:
                close_status_code = 1000
            if close_msg is None:
                close_msg = "websocket is closed"
            self._callback.on_close(close_status_code, close_msg)
        except Exception as e:
            logger.error(f"Error: {e}")

    def _on_open(self, ws):  # pylint: disable=unused-argument
        self._callback.on_connected()

    # def _on_pong(self, _):
    #     _log.debug("on pong")

    def start(self, dialog_id, enable_voice_detection=False, task_id=None):
        """
        Initialize WebSocket connection and send start request.
        :param dialog_id: Context inheritance flag. Not needed for new
            dialogs.
               If inheriting previous dialog history, record and pass
               the previous dialog_id
        :param enable_voice_detection: Whether to enable voice detection,
            optional, default False
        :param task_id: DashScope request task ID, auto-generated by
            default. You can specify this ID to track the request.
        """
        self._voice_detection = enable_voice_detection
        self._connect(self.api_key)
        logger.debug("connected with server.")
        self._send_start_request(
            dialog_id,
            self.request_params,
            task_id=task_id,
        )

    def start_speech(self):
        """Start uploading speech data"""
        _send_speech_json = self.request.generate_common_direction_request(
            "SendSpeech",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def send_audio_data(self, speech_data: bytes):
        """Send speech data"""
        self.__send_binary_frame(speech_data)

    def stop_speech(self):
        """Stop uploading speech data"""
        _send_speech_json = self.request.generate_common_direction_request(
            "StopSpeech",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def interrupt(self):
        """Request server to start speaking"""
        _send_speech_json = self.request.generate_common_direction_request(
            "RequestToSpeak",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def request_to_respond(
        self,
        request_type: str,
        text: str,
        parameters: RequestToRespondParameters = None,
    ):
        """Request server to synthesize speech from text directly"""
        _send_speech_json = self.request.generate_request_to_response_json(
            direction_name="RequestToRespond",
            dialog_id=self.dialog_id,
            request_type=request_type,
            text=text,
            parameters=parameters,
        )
        self._send_text_frame(_send_speech_json)

    @abstractmethod
    def request_to_respond_prompt(self, text):
        """Request server to reply with text response via text request"""
        return

    def local_responding_started(self):
        """Local TTS playback started"""
        _send_speech_json = self.request.generate_common_direction_request(
            "LocalRespondingStarted",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def local_responding_ended(self):
        """Local TTS playback ended"""
        _send_speech_json = self.request.generate_common_direction_request(
            "LocalRespondingEnded",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def send_heart_beat(self):
        """Send heartbeat"""
        _send_speech_json = self.request.generate_common_direction_request(
            "HeartBeat",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def update_info(self, parameters: RequestToRespondParameters = None):
        """Update information"""
        _send_speech_json = self.request.generate_update_info_json(
            direction_name="UpdateInfo",
            dialog_id=self.dialog_id,
            parameters=parameters,
        )
        self._send_text_frame(_send_speech_json)

    def stop(self):
        if self.ws is None or not self.ws.sock or not self.ws.sock.connected:
            self._callback.on_close(1001, "websocket is not connected")
            return
        _send_speech_json = self.request.generate_stop_request(
            "Stop",
            self.dialog_id,
        )
        self._send_text_frame(_send_speech_json)

    def get_dialog_state(self) -> dialog_state.DialogState:
        return self.dialog_state.get_current_state()

    def get_conversation_mode(self) -> str:
        """get mode of conversation: support tap2talk/push2talk/duplex"""
        return self.request_params.upstream.mode

    """Internal methods"""  # pylint: disable=pointless-string-statement

    def _send_start_request(
        self,
        dialog_id: str,
        request_params: RequestParameters,
        task_id: str = None,
    ):
        """Send 'Start' request"""
        _start_json = self.request.generate_start_request(
            workspace_id=self.workspace_id,
            direction_name="Start",
            dialog_id=dialog_id,
            app_id=self.app_id,
            request_params=request_params,
            model=self.model,
            task_id=task_id,
        )
        # send start request
        self._send_text_frame(_start_json)

    def _run_forever(self):
        self.ws.run_forever(ping_interval=None, ping_timeout=None)

    def _connect(self, api_key: str):
        """Initialize WebSocket connection and send startup request."""
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.request.get_websocket_header(api_key),
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self._run_forever)
        self.thread.daemon = True
        self.thread.start()

        self._wait_for_connection()

    def close(self):
        if self.ws is None or not self.ws.sock or not self.ws.sock.connected:
            return
        self.ws.close()

    def _wait_for_connection(self):
        """Wait for WebSocket connection to be established"""
        timeout = 5
        start_time = time.time()
        while (
            not (self.ws.sock and self.ws.sock.connected)
            and (time.time() - start_time) < timeout
        ):
            time.sleep(0.1)  # Brief sleep to avoid busy polling

    def _send_text_frame(self, text: str):
        logger.info(">>>>>> send text frame : %s", text)
        self.ws.send(text, websocket.ABNF.OPCODE_TEXT)

    def __send_binary_frame(self, binary: bytes):
        # _log.info('send binary frame length: %d' % len(binary))
        self.ws.send(binary, websocket.ABNF.OPCODE_BINARY)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Clean up all resources"""
        try:
            if self.ws:
                self.ws.close()
            if self.thread and self.thread.is_alive():
                # Set flag to notify thread to exit
                self.thread.join(timeout=2)
            # Clear references
            self.ws = None
            self.thread = None
            self._callback = None
            self.response = None
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")


class _Request:
    def __init__(self):
        # websocket header
        self.ws_headers = None
        # request body for voice chat
        self.header = None
        self.payload = None
        # params
        self.task_id = None
        self.app_id = None
        self.workspace_id = None

    def get_websocket_header(self, api_key):
        ua = get_user_agent()
        self.ws_headers = {
            "User-Agent": ua,
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        log_headers = self.ws_headers.copy()
        log_headers["Authorization"] = "REDACTED"
        logger.info("websocket header: %s", log_headers)
        return self.ws_headers

    def generate_start_request(
        self,
        direction_name: str,
        dialog_id: str,
        app_id: str,
        request_params: RequestParameters,
        model: str = None,
        workspace_id: str = None,
        task_id: str = None,
    ) -> str:
        """
        Build startup request data for voice chat service.
        :param app_id: Console application ID
        :param request_params: Parameters in start request body
        :param direction_name:
        :param dialog_id: Dialog ID.
        :param workspace_id: Console workspace ID, optional field.
        :param model: Model
        :param task_id: DashScope request task ID, auto-generated by
            default. You can specify this ID to track the request.
        :return: Startup request dictionary.
        """
        self.task_id = task_id
        self._get_dash_request_header(ActionType.START)
        self._get_dash_request_payload(
            direction_name,
            dialog_id,
            app_id,
            workspace_id=workspace_id,
            request_params=request_params,
            model=model,
        )

        cmd = {
            "header": self.header,
            "payload": self.payload,
        }
        return json.dumps(cmd)

    def generate_common_direction_request(
        self,
        direction_name: str,
        dialog_id: str,
    ) -> str:
        """
        Build command request data for voice chat service.
        :param direction_name: Command.
        :param dialog_id: Dialog ID.
        :return: Command request JSON.
        """
        self._get_dash_request_header(ActionType.CONTINUE)
        self._get_dash_request_payload(direction_name, dialog_id, self.app_id)
        cmd = {
            "header": self.header,
            "payload": self.payload,
        }
        return json.dumps(cmd)

    def generate_stop_request(
        self,
        direction_name: str,
        dialog_id: str,
    ) -> str:
        """
        Build stop request data for voice chat service.
        :param direction_name: Directive name
        :param dialog_id: Dialog ID.
        :return: Stop request JSON.
        """
        self._get_dash_request_header(ActionType.FINISHED)
        self._get_dash_request_payload(direction_name, dialog_id, self.app_id)

        cmd = {
            "header": self.header,
            "payload": self.payload,
        }
        return json.dumps(cmd)

    def generate_request_to_response_json(
        self,
        direction_name: str,
        dialog_id: str,
        request_type: str,
        text: str,
        parameters: RequestToRespondParameters = None,
    ) -> str:
        """
        Build command request data for voice chat service.
        :param direction_name: Command.
        :param dialog_id: Dialog ID.
        :param request_type: Interaction type the service should adopt,
            transcript means convert text to speech directly,
            prompt means send text to LLM for response
        :param text: Text.
        :param parameters: Parameters in command request body
        :return: Command request dictionary.
        """
        self._get_dash_request_header(ActionType.CONTINUE)

        custom_input = RequestToRespondBodyInput(
            app_id=self.app_id,
            directive=direction_name,
            dialog_id=dialog_id,
            type_=request_type,
            text=text,
        )

        self._get_dash_request_payload(
            direction_name,
            dialog_id,
            self.app_id,
            request_params=parameters,  # type: ignore[arg-type]
            custom_input=custom_input,
        )
        cmd = {
            "header": self.header,
            "payload": self.payload,
        }
        return json.dumps(cmd)

    def generate_update_info_json(
        self,
        direction_name: str,
        dialog_id: str,
        parameters: RequestToRespondParameters = None,
    ) -> str:
        """
        Build command request data for voice chat service.
        :param direction_name: Command.
        :param parameters: Parameters in command request body
        :return: Command request dictionary.
        """
        self._get_dash_request_header(ActionType.CONTINUE)

        custom_input = RequestToRespondBodyInput(
            app_id=self.app_id,
            directive=direction_name,
            dialog_id=dialog_id,
        )

        self._get_dash_request_payload(
            direction_name,
            dialog_id,
            self.app_id,
            request_params=parameters,  # type: ignore[arg-type]
            custom_input=custom_input,
        )
        cmd = {
            "header": self.header,
            "payload": self.payload,
        }
        return json.dumps(cmd)

    def _get_dash_request_header(self, action: str):
        """
        Build request protocol header for multimodal dialog request.
        :param action: ActionType DashScope protocol action, supports:
            run-task, continue-task, finish-task
        """
        if self.task_id is None:
            self.task_id = get_random_uuid()
        self.header = DashHeader(action=action, task_id=self.task_id).to_dict()

    def _get_dash_request_payload(
        self,
        direction_name: str,
        dialog_id: str,
        app_id: str,
        workspace_id: str = None,
        request_params: RequestParameters = None,
        custom_input=None,
        model: str = None,
    ):
        """
        Build request protocol payload for multimodal dialog request.
        :param direction_name: Internal directive name in dialog protocol
        :param dialog_id: Dialog ID.
        :param app_id: Console application ID
        :param request_params: Parameters in start request body
        :param custom_input: Custom input
        :param model: Model
        """
        if custom_input is not None:
            input = custom_input  # pylint: disable=redefined-builtin
        else:
            input = RequestBodyInput(
                workspace_id=workspace_id,
                app_id=app_id,
                directive=direction_name,
                dialog_id=dialog_id,
            )

        self.payload = DashPayload(
            model=model,
            input=input,
            parameters=request_params,
        ).to_dict()


class _Response:
    def __init__(
        self,
        state: dialog_state.StateMachine,
        callback: MultiModalCallback,
        close_callback=None,
    ):
        super().__init__()
        self.dialog_id = None  # Dialog ID
        self.dialog_state = state
        self._callback = callback
        self._close_callback = close_callback  # Save close callback function

    # pylint: disable=inconsistent-return-statements
    def handle_text_response(self, response_json: str):
        """
        Handle response data from voice chat service.
        :param response_json: Original JSON string response from server.
        """
        logger.info("<<<<<< server response: %s", response_json)
        try:
            # Attempt to parse message as JSON
            json_data = json.loads(response_json)
            if (
                "status_code" in json_data["header"]
                and json_data["header"]["status_code"] != 200
            ):
                logger.error(
                    "Server returned invalid message: %s",
                    response_json,
                )
                if self._callback:
                    self._callback.on_error(response_json)
                return
            if (
                "event" in json_data["header"]
                and json_data["header"]["event"] == "task-failed"
            ):
                logger.error(
                    "Server returned invalid message: %s",
                    response_json,
                )
                if self._callback:
                    self._callback.on_error(response_json)
                return None

            payload = json_data["payload"]
            if "output" in payload and payload["output"] is not None:
                response_event = payload["output"]["event"]
                logger.info("Server response event: %s", response_event)
                self._handle_text_response_in_conversation(
                    response_event=response_event,
                    response_json=json_data,
                )
            del json_data

        except json.JSONDecodeError:
            logger.error("Failed to parse message as JSON.")

    def _handle_text_response_in_conversation(
        self,
        response_event: str,
        response_json: dict,
    ):  # pylint: disable=too-many-branches
        payload = response_json["payload"]
        try:
            if response_event == RESPONSE_NAME_STARTED:
                self._handle_started(payload["output"])
            elif response_event == RESPONSE_NAME_STOPPED:
                self._handle_stopped()
            elif response_event == RESPONSE_NAME_STATE_CHANGED:
                self._handle_state_changed(payload["output"]["state"])
                logger.debug(
                    "service response change state: %s",
                    payload["output"]["state"],
                )
            elif response_event == RESPONSE_NAME_REQUEST_ACCEPTED:
                self._handle_request_accepted()
            elif response_event == RESPONSE_NAME_SPEECH_STARTED:
                self._handle_speech_started()
            elif response_event == RESPONSE_NAME_SPEECH_ENDED:
                self._handle_speech_ended()
            elif response_event == RESPONSE_NAME_RESPONDING_STARTED:
                self._handle_responding_started()
            elif response_event == RESPONSE_NAME_RESPONDING_ENDED:
                self._handle_responding_ended(payload)
            elif response_event == RESPONSE_NAME_SPEECH_CONTENT:
                self._handle_speech_content(payload)
            elif response_event == RESPONSE_NAME_RESPONDING_CONTENT:
                self._handle_responding_content(payload)
            elif response_event == RESPONSE_NAME_ERROR:
                self._callback.on_error(json.dumps(response_json))
            elif response_event == RESPONSE_NAME_HEART_BEAT:
                logger.debug("Server response heart beat")
            else:
                logger.error("Unknown response name: %s", response_event)
        except json.JSONDecodeError:
            logger.error("Failed to parse message as JSON.")

    def handle_binary_response(self, message: bytes):
        # logger.debug('<<<recv binary {}'.format(len(message)))
        self._callback.on_speech_audio_data(message)

    def _handle_request_accepted(self):
        self._callback.on_request_accepted()

    def _handle_started(self, payload: dict):
        self.dialog_id = payload["dialog_id"]
        self._callback.on_started(self.dialog_id)  # type: ignore[arg-type]

    def _handle_stopped(self):
        self._callback.on_stopped()
        if self._close_callback is not None:
            self._close_callback()

    def _handle_state_changed(self, state: str):
        """
        Handle voice chat state transitions.
        :param state: State.
        """
        self.dialog_state.change_state(state)
        self._callback.on_state_changed(self.dialog_state.get_current_state())

    def _handle_speech_started(self):
        self._callback.on_speech_started()

    def _handle_speech_ended(self):
        self._callback.on_speech_ended()

    def _handle_responding_started(self):
        self._callback.on_responding_started()

    def _handle_responding_ended(self, payload: dict):
        self._callback.on_responding_ended(payload)

    def _handle_speech_content(self, payload: dict):
        self._callback.on_speech_content(payload)

    def _handle_responding_content(self, payload: dict):
        self._callback.on_responding_content(payload)
