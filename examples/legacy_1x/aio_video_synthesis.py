# -*- coding: utf-8 -*-
"""AioVideoSynthesis video-generation async example (wanx series).

Demonstrates in order: blocking call -> submit an async task and
wait/fetch it -> list tasks.
Note: video-generation tasks take a while; please be patient.
"""

import asyncio
import os
from http import HTTPStatus

from dashscope.aigc.video_synthesis import AioVideoSynthesis

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "wan2.7-t2v")

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
    rsp = await AioVideoSynthesis.call(
        api_key=api_key,
        model=MODEL,
        prompt=PROMPT,
    )
    print_response(rsp)

    # Async task: submit, then wait / fetch / cancel as needed
    print("----- async_call + wait + fetch -----")
    task = await AioVideoSynthesis.async_call(
        api_key=api_key,
        model=MODEL,
        prompt=PROMPT,
    )
    print_response(task)
    if task.status_code == HTTPStatus.OK:
        task_id = task.output.task_id

        # Wait for the task to complete
        print(f"----- wait (task_id={task_id}) -----")
        print_response(await AioVideoSynthesis.wait(task_id))

        # Fetch the task status and result by task_id
        print(f"----- fetch (task_id={task_id}) -----")
        print_response(await AioVideoSynthesis.fetch(task_id))

        # Cancel the task (it is expected to fail if the task has
        # already completed)
        # print(f"----- cancel (task_id={task_id}) -----")
        # print_response(await AioVideoSynthesis.cancel(task_id))

    # List historical tasks
    print("----- list -----")
    print_response(await AioVideoSynthesis.list())


if __name__ == "__main__":
    asyncio.run(main())
