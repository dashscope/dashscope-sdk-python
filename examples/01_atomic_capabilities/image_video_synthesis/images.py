# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 image-generation example (wanx series).

Demonstrates in order: one-call generate (submit + wait) -> manual
async task submit/wait/fetch -> list tasks.
"""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "wanx2.1-t2i-turbo")

PROMPT = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"


if __name__ == "__main__":
    client = dashscope.Client()

    # One-call generate: submits the task and waits for the result
    print("----- generate (submit + wait) -----")
    job = client.images.generate(
        model=MODEL,
        prompt=PROMPT,
        n=1,
        size="1024*1024",
    )
    if job.succeeded:
        for url in job.image_urls:
            print(f"image url: {url}")
    else:
        print(f"job failed: {job.output}")

    # Manual async-task flow: submit first, then wait/fetch by task_id
    print("----- jobs.create + wait + fetch -----")
    pending = client.images.jobs.create(
        model=MODEL,
        prompt=PROMPT,
        n=1,
        size="1024*1024",
    )
    if pending.output is not None:
        task_id = pending.output[
            "task_id"
        ]  # pylint: disable=unsubscriptable-object
    else:
        raise RuntimeError("pending.output is None")
    print(f"submitted task_id: {task_id}")

    final = client.images.jobs.wait(task_id)
    print(f"wait finished, succeeded: {final.succeeded}")

    snapshot = client.images.jobs.fetch(task_id)
    print(f"fetch output: {snapshot.output}")

    # List historical tasks
    print("----- jobs.list -----")
    result = client.images.jobs.list()
    print(f"total tasks: {result.total}")
    for task in result.tasks[:3]:
        print(f"task_id={task.task_id}, status={task.status}")


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   This capability is a long-running jobs flow; reuse the
#   generic polling and failover patterns above.
# -------------------------------------------------------------
