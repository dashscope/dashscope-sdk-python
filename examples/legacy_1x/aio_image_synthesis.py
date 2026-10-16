# -*- coding: utf-8 -*-
"""AioImageSynthesis text-to-image async example (wanx series).

Demonstrates in order: blocking call -> submit an async task and
wait/fetch it -> list tasks.
"""

import asyncio
import os
from http import HTTPStatus

from dashscope.aigc.image_synthesis import AioImageSynthesis

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "wan2.2-t2i-flash")

PROMPT = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"


def print_response(rsp) -> None:
    if rsp.status_code == HTTPStatus.OK:
        print(rsp)
    else:
        print(
            f"Failed, status_code: {rsp.status_code}, "
            f"code: {rsp.code}, message: {rsp.message}",
        )


async def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")

    # Blocking wait: submit the task and block until the result
    print("----- call (wait for result) -----")
    rsp = await AioImageSynthesis.call(
        api_key=api_key,
        model=MODEL,
        prompt=PROMPT,
        n=1,
        size="1024*1024",
    )
    print_response(rsp)

    # Note: the sync_call sync API currently only supports
    # wan2.2-t2i-flash and wan2.2-t2i-plus
    # rsp = await AioImageSynthesis.sync_call(
    #     api_key=api_key, model=MODEL, prompt=PROMPT, n=1, size="1024*1024",
    # )
    # print_response(rsp)

    # Async task: submit, then wait / fetch / cancel as needed
    print("----- async_call + wait + fetch -----")
    task = await AioImageSynthesis.async_call(
        api_key=api_key,
        model=MODEL,
        prompt=PROMPT,
        n=1,
        size="1024*1024",
    )
    print_response(task)
    if task.status_code == HTTPStatus.OK:
        task_id = task.output.task_id

        # Wait for the task to complete
        print(f"----- wait (task_id={task_id}) -----")
        print_response(await AioImageSynthesis.wait(task_id))

        # Fetch the task status and result by task_id
        print(f"----- fetch (task_id={task_id}) -----")
        print_response(await AioImageSynthesis.fetch(task_id))

        # Cancel the task (it is expected to fail if the task has
        # already completed)
        # print(f"----- cancel (task_id={task_id}) -----")
        # print_response(await AioImageSynthesis.cancel(task_id))

    # List historical tasks
    print("----- list -----")
    print_response(await AioImageSynthesis.list())


if __name__ == "__main__":
    asyncio.run(main())
