# -*- coding: utf-8 -*-
import json
import sys
from http import HTTPStatus

from dashscope import Assistants, Messages, Runs, Threads


def create_assistant():
    # create assistant with information
    assistant = Assistants.create(
        model="qwen-max",  # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        name="smart helper",
        description="A tool helper.",
        instructions="You are a helpful assistant.",  # noqa E501
    )

    return assistant


def verify_status_code(res):
    if res.status_code != HTTPStatus.OK:
        print("Failed: ")
        print(res)
        sys.exit(res.status_code)


if __name__ == "__main__":
    # create assistant
    assistant = create_assistant()
    print("create assistant:\n%s\n" % assistant)
    verify_status_code(assistant)

    # create thread.
    thread = Threads.create(
        messages=[
            {
                "role": "user",
                "content": "如何做出美味的牛肉炖土豆？",
            },
        ],
    )
    print("create thread:\n%s\n" % thread)
    verify_status_code(thread)

    # create run
    run = Runs.create(thread.id, assistant_id=assistant.id)

    print(run)
    print("create run:\n%s\n" % run)

    verify_status_code(run)
    # wait for run completed or requires_action
    run_status = Runs.wait(run.id, thread_id=thread.id)
    print(run_status)
    print("run status:\n%s\n" % run_status)
    if run_status.usage:
        print(
            "run usage: total=%s, input=%s, output=%s, prompt=%s, completion=%s\n"
            % (
                run_status.usage.get("total_tokens"),
                run_status.usage.get("input_tokens"),
                run_status.usage.get("output_tokens"),
                run_status.usage.get("prompt_tokens"),
                run_status.usage.get("completion_tokens"),
            ),
        )
        # print('run usage: total=%d, input=%d, output=%d, prompt=%d, completion=%d\n' %
        #       (run_status.usage.total_tokens, run_status.usage.input_tokens, run_status.usage.output_tokens,
        #        run_status.usage.prompt_tokens, run_status.usage.completion_tokens))

    # get the thread messages.
    msgs = Messages.list(thread.id)
    print(msgs)
    print("thread messages:\n%s\n" % msgs)
    print(
        json.dumps(
            msgs,
            ensure_ascii=False,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4,
        ),
    )
