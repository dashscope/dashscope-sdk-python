# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 video-generation example (wan series).

Demonstrates in order: one-call generate (submit + wait) -> manual
async task submit/wait/fetch -> list tasks.
Note: video-generation tasks take a while; please be patient.
"""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "wan2.7-t2v")

PROMPT = "一只小猫在月光下奔跑"


if __name__ == "__main__":
    client = dashscope.Client()

    # One-call generate: submits the task and waits for the result
    print("----- generate (submit + wait) -----")
    job = client.videos.generate(model=MODEL, prompt=PROMPT)
    if job.succeeded:
        assert job.output is not None
        print(f"video url: {job.output['video_url']}")
    else:
        print(f"job failed: {job.output}")

    # Manual async-task flow: submit first, then wait/fetch by task_id
    print("----- jobs.create + wait + fetch -----")
    pending = client.videos.jobs.create(model=MODEL, prompt=PROMPT)
    if pending.output is not None:
        task_id = pending.output[
            "task_id"
        ]  # pylint: disable=unsubscriptable-object
    else:
        raise RuntimeError("pending.output is None")
    print(f"submitted task_id: {task_id}")

    final = client.videos.jobs.wait(task_id)
    print(f"wait finished, succeeded: {final.succeeded}")
    print(f"video url: {final.output.get('video_url')}")

    snapshot = client.videos.jobs.fetch(task_id)
    print(f"fetch output: {snapshot.output}")

    # List historical tasks
    print("----- jobs.list -----")
    result = client.videos.jobs.list()
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
