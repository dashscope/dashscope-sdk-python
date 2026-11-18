# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Live gated e2e for the Batch workflow (B3 full closed loop).

Uses the public, zero-inference-cost test model ``batch-test-model`` with
endpoint ``/v1/chat/ds-test`` (documented in the Bailian OpenAI-compatible
Batch guide): it skips inference and returns a fixed
``{"content": "This is a test result."}``, so the whole upload -> create ->
wait -> download_output loop can be exercised cheaply with a public model.

Skipped unless explicitly opted in. Creating batch jobs is a write side
effect, so beyond the usual key/workspace gate this requires
``DASHSCOPE_E2E_BATCH=1``::

    DASHSCOPE_API_KEY=... DASHSCOPE_WORKSPACE=... DASHSCOPE_E2E_BATCH=1 \\
        pytest tests/protocols/test_batch_e2e.py

The test model caps input at 1 MB / 100 rows and 2 parallel jobs; this case
uploads 2 rows and cleans up its files afterwards.
"""

from __future__ import annotations

import json
import os

import pytest

from dashscope import Client

_ENDPOINT = "/v1/chat/ds-test"
_MODEL = "batch-test-model"


def _opted_in() -> bool:
    return os.environ.get("DASHSCOPE_E2E_BATCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _make_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    workspace = os.environ.get("DASHSCOPE_WORKSPACE")
    if not api_key:
        pytest.skip("DASHSCOPE_API_KEY not set")
    if not workspace:
        pytest.skip("DASHSCOPE_WORKSPACE not set")
    if not _opted_in():
        pytest.skip("set DASHSCOPE_E2E_BATCH=1 to create live batch jobs")
    return Client(api_key=api_key, workspace=workspace, max_retries=1)


def _rows():
    return [
        {
            "custom_id": "1",
            "method": "POST",
            "url": _ENDPOINT,
            "body": {
                "model": _MODEL,
                "messages": [{"role": "user", "content": "你好！"}],
            },
        },
        {
            "custom_id": "2",
            "method": "POST",
            "url": _ENDPOINT,
            "body": {
                "model": _MODEL,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        },
    ]


def test_batch_full_loop_with_test_model(tmp_path):
    client = _make_client()
    input_path = tmp_path / "test_model.jsonl"
    input_path.write_text(
        "".join(json.dumps(r) + "\n" for r in _rows()),
        encoding="utf-8",
    )
    uploaded = []
    with client:
        assert client.batches.validate_input(
            str(input_path),
            endpoint=_ENDPOINT,
        ) == 2
        file_obj = client.batches.files.upload(
            file_path=str(input_path),
            purpose="batch",
        )
        uploaded.append(file_obj.id)
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=_ENDPOINT,
            completion_window="24h",
        )
        batch = client.batches.wait(
            batch,
            poll_interval=5,
            max_interval=30,
            timeout=300,
        )
        assert batch.status == "completed", batch.model_dump()
        try:
            content = client.batches.download_output(batch)
            assert content is not None
            assert "This is a test result." in content.text
            rows = [json.loads(line) for line in content.text.splitlines()]
            assert {r["custom_id"] for r in rows} == {"1", "2"}
            for row in rows:
                assert row["response"]["status_code"] == 200
            if batch.output_file_id:
                uploaded.append(batch.output_file_id)
            error_content = client.batches.download_error(batch)
            # A clean test-model batch has no failed rows.
            assert error_content is None or isinstance(error_content, bytes)
            if batch.error_file_id:
                uploaded.append(batch.error_file_id)
        finally:
            for file_id in uploaded:
                try:
                    client.batches.files.delete(file_id)
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass