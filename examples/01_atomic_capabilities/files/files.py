# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 file-management example: upload -> list -> get ->
delete.

Creates a small temporary JSONL file locally so the example is
self-contained.
"""

import os
import tempfile

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").


if __name__ == "__main__":
    client = dashscope.Client()

    # Build a small local file to upload
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
    ) as f:
        f.write('{"prompt": "你好", "completion": "你好！有什么可以帮你？"}\n')
        file_path = f.name

    try:
        # Upload
        print("----- upload file -----")
        uploaded = client.files.upload(
            file_path=file_path,
            purpose="fine-tune",
            description="example upload",
        )
        print(f"file_id: {uploaded.file_id}, name: {uploaded.name}")

        # List files
        print("----- list files -----")
        files = client.files.list()
        for item in files.files[:5]:
            print(f"file_id={item.file_id}, name={item.name}")

        # Get one file
        print("----- get file -----")
        item = client.files.get(uploaded.file_id)
        print(f"file_id={item.file_id}, name={item.name}")

        # Delete the uploaded file
        print("----- delete file -----")
        deleted = client.files.delete(uploaded.file_id)
        print(f"deleted: {deleted.deleted}")
    finally:
        os.unlink(file_path)


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   No dedicated harness for this capability yet; reuse the
#   generic patterns above for fault-tolerant upload/delete retries.
# -------------------------------------------------------------
