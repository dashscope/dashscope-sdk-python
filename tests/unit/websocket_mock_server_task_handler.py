# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import json

import aiohttp

from dashscope.common.error import UnexpectedMessageReceived
from dashscope.protocol.websocket import (
    ACTION_KEY,
    EVENT_KEY,
    ActionType,
    EventType,
    WebsocketStreamingMode,
)


class WebSocketTaskProcessor:
    """WebSocket state machine."""

    def __init__(
        self,
        ws,
        task_id,
        streaming_mode,
        model,
        task,
        is_binary_in,
        is_binary_out,
        run_task_json_message,
    ) -> None:
        self.ws = ws
        self.error = ""
        self.error_message = ""
        self.task_id = task_id
        self.model = model
        self.task = task
        self.streaming_mode = streaming_mode
        self.run_task_json_message = run_task_json_message
        self.is_binary_in = is_binary_in
        self.is_binary_out = is_binary_out
        self._duplex_task_finished = False

    async def aio_call(self):  # pylint: disable=too-many-branches
        # no matter what, send start event first.
        await self._send_start_event()
        if self.streaming_mode == WebsocketStreamingMode.NONE:
            # if binary data, we need to receive data
            if self.is_binary_in:
                binary_data = (
                    await self._receive_batch_binary()
                )  # ignore timeout.
                print(f"Receive binary data, length: {len(binary_data)}")
            # send "event":"task-finished"
            if self.is_binary_out:
                # send binary data
                await self.send_batch_streaming_output()
                await self._send_task_finished(payload={})
            elif self.is_binary_in:
                return await self._send_task_finished(
                    payload={
                        "output": {
                            "text": "world",
                        },
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 200,
                        },
                    },
                )  # binary input, result with world.
            else:
                await self._send_task_finished(
                    payload={
                        "output": {
                            "text": self.run_task_json_message["payload"][
                                "input"
                            ]["prompt"],
                        },
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 200,
                        },
                    },
                )  # for echo message out.

        elif self.streaming_mode == WebsocketStreamingMode.IN:
            if self.is_binary_in:  # binary data
                await self._receive_streaming_binary_data()
            else:
                await self._receive_streaming_text_data()
            # processing data
            if self.is_binary_out:
                await self.send_batch_streaming_output()
                await self._send_task_finished(payload={})
            else:
                if self.is_binary_in:
                    return await self._send_task_finished(
                        payload={
                            "output": {
                                "text": "world",
                            },
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 200,
                            },
                        },
                    )
                else:
                    await self._send_task_finished(
                        payload={
                            "output": {
                                "text": "world",
                            },
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 200,
                            },
                        },
                    )
        elif self.streaming_mode == WebsocketStreamingMode.OUT:
            if self.is_binary_in:  # run task without data, data is binary.
                binary_data = await self._receive_batch_binary()
                print(len(binary_data))
            else:
                pass  # batch data is in run-task
            # processing data
            if self.is_binary_out:
                await self.send_streaming_binary_output()
            else:
                await self.send_streaming_text_output()

            await self._send_task_finished(payload={})

        else:  # duplex mode
            if self.is_binary_in:
                send_task = asyncio.create_task(
                    self._receive_streaming_binary_data(),
                )
            else:
                send_task = asyncio.create_task(
                    self._receive_streaming_text_data(),
                )
            if self.is_binary_out:
                receive_task = asyncio.create_task(
                    self.send_streaming_binary_output(),
                )
            else:
                receive_task = asyncio.create_task(
                    self.send_streaming_text_output(),
                )

            _, _ = await asyncio.gather(receive_task, send_task)

            await self._send_task_finished(payload={})

    async def send_streaming_binary_output(self):
        for _ in range(10):
            data = bytes([0x01] * 100)
            await self.ws.send_bytes(data)

    async def send_streaming_text_output(self):
        headers = {
            "task_id": self.task_id,
            "event": "result-generated",
        }
        for _ in range(10):
            payload = {
                "output": {
                    "text": "world",
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                },
            }
            msg = self._build_up_message(headers=headers, payload=payload)
            await self.ws.send_str(msg)
        print("send_streaming_text_output finished!")

    async def send_batch_streaming_output(self):
        data = bytes([0x01] * 100)
        await self.ws.send_bytes(data)

    async def _send_start_event(self):
        headers = {"task_id": self.task_id, EVENT_KEY: EventType.STARTED}
        payload = {}
        message = self._build_up_message(headers, payload=payload)
        print(f"sending task started event message: {message}")
        await self.ws.send_str(message)

    async def _send_task_finished(self, payload):
        headers = {"task_id": self.task_id, EVENT_KEY: EventType.FINISHED}
        message = self._build_up_message(headers, payload)
        print(f"sending task finished message: {message}")
        await self.ws.send_str(message)

    async def _receive_streaming_binary_data(self):
        while True:
            msg = await self.ws.receive()
            if await self.validate_message(msg):
                return
            if msg.type == aiohttp.WSMsgType.BINARY:
                # real server need return data and process.
                print(f"Receive binary data length: {len(msg.data)}")
            elif msg.type == aiohttp.WSMsgType.TEXT:
                req = msg.json()
                print(f"Receive {req['header'][ACTION_KEY]} event")
                if req["header"][ACTION_KEY] == ActionType.FINISHED:
                    self._duplex_task_finished = True
                    break
                print(f"Unknown message: {msg}")
            else:
                raise UnexpectedMessageReceived(
                    f"Expect binary data but receive {msg.type}!",
                )

    async def _receive_streaming_text_data(self):
        payload = []
        payload.append(self.run_task_json_message["payload"]["input"])
        while True:
            msg = await self.ws.receive()
            if await self.validate_message(msg):
                return
            if msg.type == aiohttp.WSMsgType.TEXT:
                msg_json = msg.json()
                print(f"Receive {msg_json['header'][ACTION_KEY]} event")
                if msg_json["header"][ACTION_KEY] == ActionType.CONTINUE:
                    print(f"Receive text data: {msg_json['payload']}")
                    payload.append(msg_json["payload"])
                elif msg_json["header"][ACTION_KEY] == ActionType.FINISHED:
                    print(f"Receive text data: {msg_json['payload']}")
                    if msg_json["payload"]:
                        payload.append(msg_json["payload"])
                    self._duplex_task_finished = True
                    return payload
                print(f"Unknown message: {msg_json}")
            else:
                raise UnexpectedMessageReceived(
                    f"Expect binary data but receive {msg.type}!",
                )

    async def _receive_batch_binary(self):
        """If the data is not binary, data is send in start package.
        otherwise data is in data package.
        It is assumed that the client will send the end command. we send.

        Returns:
            No:
        """
        while True:
            msg = await self.ws.receive()
            if await self.validate_message(msg):
                break
            if msg.type == aiohttp.WSMsgType.BINARY:
                return msg.data
            raise UnexpectedMessageReceived(
                f"Expect binary data but receive {msg.type}!",
            )

    async def _receive_batch_text(self):
        """If the data is not binary, data is send in start package.
        otherwise data is in data package.
        It is assumed that the client will send the end command. we send.

        Returns:
            No:
        """
        final_data = self.run_task_json_message["payload"]
        while True:
            msg = await self.ws.receive()
            if await self.validate_message(msg):
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                req = msg.json()
                print(f"Receive {req['header'][ACTION_KEY]} event")
                if req["header"][ACTION_KEY] == ActionType.START:
                    print("receive start task event")
                elif req["header"][ACTION_KEY] == ActionType.FINISHED:
                    # client is finished, send finished task binary response.
                    await self._send_task_finished(final_data)
                    break
                else:
                    print(f"Unknown message: {msg}")
            else:
                raise UnexpectedMessageReceived(f"Expect text {msg.type}!")

    def _build_up_message(self, headers, payload):
        message = {"header": headers, "payload": payload}
        return json.dumps(message)

    async def validate_message(self, msg):
        if msg.type == aiohttp.WSMsgType.CLOSED:
            print("Client close the connection")
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print(f"Connection error: {msg.data}")
            return True
        return False
