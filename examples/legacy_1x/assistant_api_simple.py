# -*- coding: utf-8 -*-
"""Assistant end-to-end conversation example: create assistant
-> thread -> run -> get reply."""

import sys
from http import HTTPStatus

from dashscope import (  # pylint: disable=no-name-in-module
    Assistants,
    Messages,
    Runs,
    Threads,
)


def verify_status_code(res):
    if res.status_code != HTTPStatus.OK:
        print(f"Failed: {res}")
        sys.exit(res.status_code)


if __name__ == "__main__":
    # Create an assistant
    print("----- create assistant -----")
    assistant = Assistants.create(
        model="qwen3.8-max",  # change the model name as needed
        name="smart helper",
        description="A tool helper.",
        instructions="You are a helpful assistant.",
    )
    print(f"assistant_id={assistant.id}")
    verify_status_code(assistant)

    # Create a thread with the first user message
    print("----- create thread -----")
    thread = Threads.create(
        messages=[
            {
                "role": "user",
                "content": "如何做出美味的牛肉炖土豆？",
            },
        ],
    )
    print(f"thread_id={thread.id}")
    verify_status_code(thread)

    # Start a run and wait for it to complete
    print("----- create run -----")
    run = Runs.create(thread.id, assistant_id=assistant.id)
    verify_status_code(run)

    run_status = Runs.wait(run.id, thread_id=thread.id)
    print(f"run status: {run_status.status}")
    if run_status.usage:
        usage = run_status.usage
        print(
            f"run usage: total={usage.get('total_tokens')}, "
            f"input={usage.get('input_tokens')}, "
            f"output={usage.get('output_tokens')}",
        )

    # List the messages in the thread (including the assistant's reply)
    print("----- list thread messages -----")
    msgs = Messages.list(thread.id)
    for message in msgs.data:
        print(f"{message['role']}: {message['content']}")
