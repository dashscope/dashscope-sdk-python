# -*- coding: utf-8 -*-
import asyncio
from http import HTTPStatus
import os
from dashscope.aigc.image_synthesis import AioImageSynthesis

model = "wan2.2-t2i-flash"
prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
task_id = "a4eee73f-2bd2-4c1c-9990-xxxxxxx"


async def __call():
    rsp = await AioImageSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        prompt=prompt,
        n=1,
        size="1024*1024",
    )
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __async_call():
    rsp = await AioImageSynthesis.async_call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        prompt=prompt,
        n=1,
        size="1024*1024",
    )
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __sync_call():
    """
    Note: This method currently now only supports wan2.2-t2i-flash and wan2.2-t2i-plus.
        Using other models will result in an error，More raw image models may be added for use later
    """
    rsp = await AioImageSynthesis.sync_call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        prompt=prompt,
        n=1,
        size="1024*1024",
    )
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __wait():
    rsp = await AioImageSynthesis.wait(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __cancel():
    rsp = await AioImageSynthesis.cancel(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __fetch():
    rsp = await AioImageSynthesis.fetch(task_id)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


async def __list():
    rsp = await AioImageSynthesis.list()
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            "sync_call Failed, status_code: %s, code: %s, message: %s"
            % (rsp.status_code, rsp.code, rsp.message),
        )


# asyncio.run(__call())
# asyncio.run(__async_call())
# asyncio.run(__sync_call())
# asyncio.run(__wait())
# asyncio.run(__cancel())
# asyncio.run(__fetch())
asyncio.run(__list())
