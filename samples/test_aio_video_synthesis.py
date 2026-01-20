# -*- coding: utf-8 -*-
import asyncio
import threading
from http import HTTPStatus
import os
from dashscope.aigc.video_synthesis import AioVideoSynthesis

prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
task_id = "a4eee73f-2bd2-4c1c-9990-xxxxxxx"
model = "wanx2.1-t2v-turbo"


async def __call():
    rsp = await AioVideoSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        prompt=prompt,
    )
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __async_call():
    rsp = await AioVideoSynthesis.async_call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        prompt=prompt,
    )
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __wait():
    rsp = await AioVideoSynthesis.wait(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __cancel():
    rsp = await AioVideoSynthesis.cancel(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __fetch():
    rsp = await AioVideoSynthesis.fetch(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __list():
    rsp = await AioVideoSynthesis.list(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


# asyncio.run(__call())
# asyncio.run(__async_call())
# asyncio.run(__wait())
# asyncio.run(__cancel())
# asyncio.run(__fetch())
asyncio.run(__list())
