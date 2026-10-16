# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 fine-tuning example: list jobs, and (optionally)
the full create -> wait -> cancel/delete lifecycle.

The read-only list/get sections run as-is. The creation section is
commented out: uncomment it after uploading a training file (see
files.py) and setting DASHSCOPE_TRAINING_FILE_ID.
"""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-turbo")
TRAINING_FILE_ID = os.environ.get("DASHSCOPE_TRAINING_FILE_ID", "")


if __name__ == "__main__":
    client = dashscope.Client()

    # List fine-tuning jobs (paged)
    print("----- list fine-tunes -----")
    page = client.fine_tunes.list(page_no=1, page_size=10)
    print(f"total: {page.total_count}")
    for job in page.jobs:
        print(
            f"job_id={job.job_id}, status={job.status}, "
            f"model={job.finetuned_model}",
        )

    # Get one job by ID (uses the first job of the list above)
    if page.jobs:
        print("----- get fine-tune -----")
        job = client.fine_tunes.get(page.jobs[0].job_id)
        print(
            f"job_id={job.job_id}, status={job.status}, "
            f"finished={job.finished}",
        )

    # Full lifecycle: create -> wait -> delete. Uncomment after
    # setting DASHSCOPE_TRAINING_FILE_ID to an uploaded file ID.
    #
    # print("----- create fine-tune -----")
    # job = client.fine_tunes.create(
    #     model=MODEL,
    #     training_file_ids=TRAINING_FILE_ID,
    #     mode="sft",
    #     hyper_parameters={"n_epochs": 2},
    #     finetuned_model="my-finetuned-model",
    # )
    # print(f"created job_id={job.job_id}")
    #
    # print("----- wait fine-tune -----")
    # final = client.fine_tunes.wait(job)
    # print(f"status={final.status}, succeeded={final.succeeded}")
    #
    # print("----- delete fine-tune -----")
    # client.fine_tunes.delete(job.job_id)


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   This capability is a long-running jobs flow; reuse the
#   generic patterns above for poll timeouts and retries.
# -------------------------------------------------------------
