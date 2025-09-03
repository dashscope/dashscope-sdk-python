# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List, Union

import dashscope
from dashscope.api_entities.dashscope_response import RecognitionResponse
from dashscope.client.base_api import BaseApi
from dashscope.common.error import (InputRequired, InvalidParameter, ModelRequired)
import websocket

from dashscope.common.logging import logger
from dashscope.protocol.websocket import ActionType


class TingWuRealtimeCallback:
    """An interface that defines callback methods for getting TingWu results.
       Derive from this class and implement its function to provide your own data.
    """

    def on_open(self) -> None:
        pass

    def on_started(self, task_id: str) -> None:
        pass

    def on_speech_listen(self, result: dict):
        pass

    def on_recognize_result(self, result: dict):
        pass

    def on_ai_result(self, result: dict):
        pass

    def on_stopped(self) -> None:
        pass

    def on_error(self, error_code: str, error_msg: str) -> None:
        pass

    def on_close(self, close_status_code, close_msg):
        """
        连接关闭时调用此方法。

        :param close_status_code: 关闭状态码
        :param close_msg: 关闭消息
        """
        pass


class TingWuRealtime(BaseApi):
    """TingWuRealtime interface.

    Args:
        model (str): The requested model_id.
        callback (TingWuRealtimeCallback): A callback that returns
            speech recognition results.
        appId (str): The dashscope tingwu app id.
        format (str): The input audio format for TingWu request.
        sample_rate (int): The input audio sample rate.
        terminology (str): The correct instruction set id.
        workspace (str): The dashscope workspace id.

        **kwargs:
            max_end_silence (int): The maximum end silence time.
            other_params (dict, `optional`): Other parameters.

    Raises:
        InputRequired: Input is required.
    """

    SILENCE_TIMEOUT_S = 60

    def __init__(self,
                 model: str,
                 callback: TingWuRealtimeCallback,
                 audio_format: str = "pcm",
                 sample_rate: int = 16000,
                 max_end_silence: int = None,
                 app_id: str = None,
                 terminology: str = None,
                 workspace: str = None,
                 api_key: str = None,
                 base_address: str = None,
                 data_id: str = None,
                 **kwargs):
        if api_key is None:
            self.api_key = dashscope.api_key
        else:
            self.api_key = api_key
        if base_address is None:
            self.base_address = dashscope.base_websocket_api_url
        else:
            self.base_address = base_address

        if model is None:
            raise ModelRequired('Model is required!')

        self.data_id = data_id
        self.max_end_silence = max_end_silence
        self.model = model
        self.audio_format = audio_format
        self.app_id = app_id
        self.terminology = terminology
        self.sample_rate = sample_rate
        # continuous recognition with start() or once recognition with call()
        self._recognition_once = False
        self._callback = callback
        self._running = False
        self._stream_data = Queue()
        self._worker = None
        self._silence_timer = None
        self._kwargs = kwargs
        self._workspace = workspace
        self._start_stream_timestamp = -1
        self._first_package_timestamp = -1
        self._stop_stream_timestamp = -1
        self._on_complete_timestamp = -1
        self.request_id_confirmed = False
        self.last_request_id = uuid.uuid4().hex
        self.request = _Request()
        self.response = _TingWuResponse(self._callback, self.close)  # 传递 self.close 作为回调

    def _on_message(self, ws, message):
        logger.debug(f"<<<<<<< Received message: {message}")
        if isinstance(message, str):
            self.response.handle_text_response(message)
        elif isinstance(message, (bytes, bytearray)):
            self.response.handle_binary_response(message)

    def _on_error(self, ws, error):
        logger.error(f"Error: {error}")
        if self._callback:
            self._callback.on_error(error_code="10", error_msg=str(error))

    def _on_close(self, ws, close_status_code, close_msg):
        try:
            logger.debug(
                "WebSocket connection closed with status {} and message {}".format(close_status_code, close_msg))
            if close_status_code is None:
                close_status_code = 1000
            if close_msg is None:
                close_msg = "websocket is closed"
            self._callback.on_close(close_status_code, close_msg)
        except Exception as e:
            logger.error(f"Error: {e}")

    def _on_open(self, ws):
        self._callback.on_open()
        self._running = True

    # def _on_pong(self):
    #     logger.debug("on pong")

    def start(self, **kwargs):
        """
        初始化WebSocket连接并发送启动请求
        """
        assert self._callback is not None, 'Please set the callback to get the TingWu result.'  # noqa E501

        if self._running:
            raise InvalidParameter('TingWu client has started.')

        # self._start_stream_timestamp = -1
        # self._first_package_timestamp = -1
        # self._stop_stream_timestamp = -1
        # self._on_complete_timestamp = -1
        if self._kwargs is not None and len(self._kwargs) != 0:
            self._kwargs.update(**kwargs)

        self._connect(self.api_key)
        logger.debug("connected with server.")
        self._send_start_request()

    def send_audio_data(self, speech_data: bytes):
        """发送语音数据"""
        self.__send_binary_frame(speech_data)

    def stop(self):
        if self.ws is None or not self.ws.sock or not self.ws.sock.connected:
            self._callback.on_close(1001, "websocket is not connected")
            return
        _send_speech_json = self.request.generate_stop_request("stop")
        self._send_text_frame(_send_speech_json)

    """内部方法"""

    def _send_start_request(self):
        """发送'Start'请求"""
        _start_json = self.request.generate_start_request(
            workspace_id=self._workspace,
            direction_name="start",
            app_id=self.app_id,
            model=self.model,
            audio_format=self.audio_format,
            sample_rate=self.sample_rate,
            terminology=self.terminology,
            max_end_silence=self.max_end_silence,
            data_id=self.data_id,
            **self._kwargs
        )
        # send start request
        self._send_text_frame(_start_json)

    def _run_forever(self):
        self.ws.run_forever(ping_interval=5, ping_timeout=4)

    def _connect(self, api_key: str):
        """初始化WebSocket连接并发送启动请求。"""
        self.ws = websocket.WebSocketApp(self.base_address, header=self.request.get_websocket_header(api_key),
                                         on_open=self._on_open,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close)
        self.thread = threading.Thread(target=self._run_forever)
        self.ws.ping_interval = 3
        self.thread.daemon = True
        self.thread.start()

        self._wait_for_connection()

    def close(self):
        if self.ws is None or not self.ws.sock or not self.ws.sock.connected:
            return
        self.ws.close()

    def _wait_for_connection(self):
        """等待WebSocket连接建立"""
        timeout = 5
        start_time = time.time()
        while not (self.ws.sock and self.ws.sock.connected) and (time.time() - start_time) < timeout:
            time.sleep(0.1)  # 短暂休眠，避免密集轮询

    def _send_text_frame(self, text: str):
        logger.info('>>>>>> send text frame : %s' % text)
        self.ws.send(text, websocket.ABNF.OPCODE_TEXT)

    def __send_binary_frame(self, binary: bytes):
        # _log.info('send binary frame length: %d' % len(binary))
        self.ws.send(binary, websocket.ABNF.OPCODE_BINARY)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """清理所有资源"""
        try:
            if self.ws:
                self.ws.close()
            if self.thread and self.thread.is_alive():
                # 设置标志位通知线程退出
                self.thread.join(timeout=2)
            # 清除引用
            self.ws = None
            self.thread = None
            self._callback = None
            self.response = None
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def send_audio_frame(self, buffer: bytes):
        """Push audio to server

        Raises:
            InvalidParameter: Cannot send data to an uninitiated recognition.
        """
        if self._running is False:
            raise InvalidParameter('TingWu client has stopped.')

        if self._start_stream_timestamp < 0:
            self._start_stream_timestamp = time.time() * 1000
        logger.debug('send_audio_frame: {}'.format(len(buffer)))
        self.__send_binary_frame(buffer)


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
        ua = 'dashscope/%s; python/%s; platform/%s; processor/%s' % (
            '1.18.0',  # dashscope version
            platform.python_version(),
            platform.platform(),
            platform.processor(),
        )
        self.ws_headers = {
            "User-Agent": ua,
            "Authorization": f"bearer {api_key}",
            "Accept": "application/json"
        }
        logger.info('websocket header: {}'.format(self.ws_headers))
        return self.ws_headers

    def generate_start_request(self, direction_name: str,
                               app_id: str,
                               model: str = None,
                               workspace_id: str = None,
                               audio_format: str = None,
                               sample_rate: int = None,
                               terminology: str = None,
                               max_end_silence: int = None,
                               data_id: str = None,
                               **kwargs
                               ) -> str:
        """
        构建语音聊天服务的启动请求数据.
        :param app_id: 管控台应用id
        :param direction_name:
        :param workspace_id: 管控台工作空间id, 非必填字段。
        :param model: 模型
        :param audio_format: 语音格式
        :param sample_rate: 采样率
        :param terminology: 术语
        :param max_end_silence: 最大静音时长
        :param data_id: 数据id
        :return: 启动请求字典.

        Args:
            :
        """
        self._get_dash_request_header(ActionType.START)
        parameters = self._get_start_parameters(audio_format=audio_format, sample_rate=sample_rate,
                                                max_end_silence=max_end_silence,
                                                terminology=terminology,
                                                **kwargs)
        self._get_dash_request_payload(direction_name=direction_name, app_id=app_id, workspace_id=workspace_id,
                                       model=model,
                                       data_id=data_id,
                                       request_params=parameters)

        cmd = {
            "header": self.header,
            "payload": self.payload
        }
        return json.dumps(cmd)

    @staticmethod
    def _get_start_parameters(audio_format: str = None,
                              sample_rate: int = None,
                              terminology: str = None,
                              max_end_silence: int = None,
                              **kwargs):
        """
        构建语音聊天服务的启动请求数据.
        :param kwargs: 启动请求body中的parameters
        :return: 启动请求字典.
        """
        parameters = {}
        if audio_format is not None:
            parameters['format'] = audio_format
        if sample_rate is not None:
            parameters['sampleRate'] = sample_rate
        if terminology is not None:
            parameters['terminology'] = terminology
        if max_end_silence is not None:
            parameters['maxEndSilence'] = max_end_silence
        if kwargs is not None and len(kwargs) != 0:
            parameters.update(kwargs)
        return parameters

    def generate_stop_request(self, direction_name: str) -> str:
        """
        构建语音聊天服务的启动请求数据.
        :param direction_name:指令名称
        :return: 启动请求json.
        """
        self._get_dash_request_header(ActionType.FINISHED)
        self._get_dash_request_payload(direction_name, self.app_id)

        cmd = {
            "header": self.header,
            "payload": self.payload
        }
        return json.dumps(cmd)

    def _get_dash_request_header(self, action: str):
        """
        构建多模对话请求的请求协议Header
        :param action: ActionType 百炼协议action 支持：run-task, continue-task, finish-task
        """
        if self.task_id is None:
            self.task_id = get_random_uuid()
        self.header = DashHeader(action=action, task_id=self.task_id).to_dict()

    def _get_dash_request_payload(self, direction_name: str,
                                  app_id: str,
                                  workspace_id: str = None,
                                  custom_input=None,
                                  model: str = None,
                                  data_id: str = None,
                                  request_params=None,
                                  ):
        """
        构建多模对话请求的请求协议payload
        :param direction_name: 对话协议内部的指令名称
        :param app_id: 管控台应用id
        :param request_params: start请求body中的parameters
        :param custom_input: 自定义输入
        :param data_id: 数据id
        :param model: 模型
        """
        if custom_input is not None:
            input = custom_input
        else:
            input = RequestBodyInput(
                workspace_id=workspace_id,
                app_id=app_id,
                directive=direction_name,
                data_id=data_id
            )

        self.payload = DashPayload(
            model=model,
            input=input.to_dict(),
            parameters=request_params
        ).to_dict()


class _TingWuResponse:
    def __init__(self, callback: TingWuRealtimeCallback, close_callback=None):
        super().__init__()
        self.task_id = None  # 对话ID.
        self._callback = callback
        self._close_callback = close_callback  # 保存关闭回调函数

    def handle_text_response(self, response_json: str):
        """
        处理语音聊天服务的响应数据.
        :param response_json: 从服务接收到的原始JSON字符串响应。
        """
        logger.info("<<<<<< server response: %s" % response_json)
        try:
            # 尝试将消息解析为JSON
            json_data = json.loads(response_json)
            if "event" in json_data["header"] and json_data["header"]["event"] == "task-failed":
                logger.error("Server returned invalid message: %s" % response_json)
                if self._callback:
                    self._callback.on_error(error_code=json_data["header"]["error_code"],
                                            error_msg=json_data["header"]["error_message"])
                return
            if "event" in json_data["header"] and json_data["header"]["event"] == "task-started":
                self._handle_started(json_data["header"]["task_id"])
                return

            payload = json_data["payload"]
            if "output" in payload and payload["output"] is not None:
                action = payload["output"]["action"]
                logger.info("Server response action: %s" % action)
                self._handle_tingwu_agent_text_response(action=action, response_json=json_data)
            del json_data

        except json.JSONDecodeError:
            logger.error("Failed to parse message as JSON.")

    def _handle_tingwu_agent_text_response(self, action: str, response_json: dict):
        payload = response_json["payload"]
        try:
            if action == "task-failed":
                self._callback.on_error(error_code=payload["output"]["errorCode"],
                                        error_msg=payload["output"]["errorMessage"])
            elif action == "speech-listen":
                self._callback.on_speech_listen(response_json)
            elif action == "recognize-result":
                self._callback.on_recognize_result(response_json)
            elif action == "ai-result":
                self._callback.on_ai_result(response_json)
            elif action == "speech-end":  # ai-result事件永远会先于speech-end事件
                self._callback.on_stopped()
                if self._close_callback is not None:
                    self._close_callback()
            else:
                logger.error("Unknown response name: {}", action)
        except json.JSONDecodeError:
            logger.error("Failed to parse message as JSON.")

    def _handle_started(self, task_id: str):
        self.task_id = task_id
        self._callback.on_started(self.task_id)


def get_random_uuid() -> str:
    """生成并返回32位UUID字符串"""
    return uuid.uuid4().hex


@dataclass
class RequestBodyInput():
    app_id: str
    directive: str
    data_id: str = field(default=None)
    workspace_id: str = field(default=None)

    def to_dict(self):
        body_input = {
            "appId": self.app_id,
            "directive": self.directive,
        }
        if self.workspace_id is not None:
            body_input["workspace_id"] = self.workspace_id
        if self.data_id is not None:
            body_input["dataId"] = self.data_id
        return body_input


@dataclass
class DashHeader:
    action: str
    task_id: str = field(default=get_random_uuid())
    streaming: str = field(default="duplex")  # 默认为 duplex

    def to_dict(self):
        return {
            "action": self.action,
            "task_id": self.task_id,
            "request_id": self.task_id,
            "streaming": self.streaming
        }


@dataclass
class DashPayload:
    task_group: str = field(default="aigc")
    function: str = field(default="generation")
    model: str = field(default="")
    task: str = field(default="multimodal-generation")
    parameters: dict = field(default=None)
    input: dict = field(default=None)

    def to_dict(self):
        payload = {
            "task_group": self.task_group,
            "function": self.function,
            "model": self.model,
            "task": self.task,
        }

        if self.parameters is not None:
            payload["parameters"] = self.parameters

        if self.input is not None:
            payload["input"] = self.input

        return payload
