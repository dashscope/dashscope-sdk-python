# -*- coding: utf-8 -*-
"""Regression tests for dataset upload hardening.

Covers:
- Issue 9: ``Datasets.upload_datasets()`` with default (None) parameters
  must not produce ``[None]`` / ``FileSpec(path=None)`` ValidationErrors.
- Issue 10: ``to_bailian_data`` and the upload wrappers must preserve the
  specific ``AgenticRLError`` classification instead of flattening every
  failure into a generic OutputError(4034) / OSSUploadError.
"""
import os
from unittest.mock import AsyncMock, patch

import pytest

# Import from the canonical application core: the legacy 1.x facade
# re-exports a distinct (subclass) error family, while the code under
# test raises the v2 classes.
from dashscope.applications.finetune.reinforcement import (
    Datasets,
    FileSpec,
    InputError,
    OutputError,
    OSSUploadError,
)
from dashscope.applications.finetune.reinforcement.common.model import (
    _normalize_file_param,
)
from dashscope.applications.finetune.reinforcement.common.utils import (
    to_bailian_data,
)


@pytest.fixture(autouse=True)
def api_key_env():
    original = os.environ.get("DASHSCOPE_API_KEY")
    os.environ["DASHSCOPE_API_KEY"] = "test-key"
    yield
    if original is None:
        os.environ.pop("DASHSCOPE_API_KEY", None)
    else:
        os.environ["DASHSCOPE_API_KEY"] = original


class TestNormalizeFileParam:
    def test_none_becomes_empty_list(self):
        assert _normalize_file_param(None) == []

    def test_string_becomes_single_item_list(self):
        assert _normalize_file_param("train.jsonl") == ["train.jsonl"]

    def test_list_is_preserved(self):
        assert _normalize_file_param(["a.jsonl", "b.jsonl"]) == [
            "a.jsonl",
            "b.jsonl",
        ]

    def test_tuple_is_converted(self):
        assert _normalize_file_param(("a.jsonl",)) == ["a.jsonl"]


class TestDatasetsUploadDefaults:
    @pytest.mark.asyncio
    async def test_all_default_params_do_not_raise(self):
        """upload_datasets() with no arguments returns (None, None)
        instead of building FileSpec(path=None)."""
        result = await Datasets.upload_datasets()
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_single_string_is_wrapped_once(self):
        """A plain string must become one FileSpec, not one per char."""
        with patch(
            "dashscope.applications.finetune.reinforcement.common.model"
            ".to_bailian_data",
            new=AsyncMock(return_value=["file-1"]),
        ) as mock_upload:
            train_ids, val_ids = await Datasets.upload_datasets(
                training_files="train.jsonl",
                validation_files="val.jsonl",
            )

        assert train_ids == ["file-1"]
        assert val_ids == ["file-1"]
        for call in mock_upload.await_args_list:
            specs = call.args[0]
            assert specs == [FileSpec(path=specs[0].path)]
            assert specs[0].path in ("train.jsonl", "val.jsonl")

    @pytest.mark.asyncio
    async def test_specific_error_is_not_rewrapped(self):
        """An InputError from the upload must keep its own code."""
        original = InputError("File too large", error_code=4035)
        with patch(
            "dashscope.applications.finetune.reinforcement.common.model"
            ".to_bailian_data",
            new=AsyncMock(side_effect=original),
        ):
            with pytest.raises(InputError) as exc_info:
                await Datasets.upload_datasets(training_files=["big.jsonl"])
        assert exc_info.value is original

    @pytest.mark.asyncio
    async def test_unexpected_error_is_wrapped_as_oss_upload_error(self):
        with patch(
            "dashscope.applications.finetune.reinforcement.common.model"
            ".to_bailian_data",
            new=AsyncMock(side_effect=ValueError("boom")),
        ):
            with pytest.raises(OSSUploadError) as exc_info:
                await Datasets.upload_datasets(training_files=["t.jsonl"])
        assert exc_info.value.error_code == 2062


class TestToBailianDataErrorClassification:
    @pytest.mark.asyncio
    async def test_no_valid_files_keeps_input_error_4031(self, tmp_path):
        missing = tmp_path / "does-not-exist.jsonl"
        with pytest.raises(InputError) as exc_info:
            await to_bailian_data([FileSpec(path=str(missing))])
        assert exc_info.value.error_code == 4031

    @pytest.mark.asyncio
    async def test_failed_status_keeps_output_error_4032(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"prompt": "hi"}', encoding="utf-8")

        with patch(
            "dashscope.applications.finetune.reinforcement.common.utils"
            ".async_http_request",
            new=AsyncMock(return_value={"status": {"code": 500}}),
        ):
            with pytest.raises(OutputError) as exc_info:
                await to_bailian_data([FileSpec(path=str(data_file))])
        assert exc_info.value.error_code == 4032

    @pytest.mark.asyncio
    async def test_partial_upload_keeps_output_error_4033(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"prompt": "hi"}', encoding="utf-8")

        with patch(
            "dashscope.applications.finetune.reinforcement.common.utils"
            ".async_http_request",
            new=AsyncMock(
                return_value={
                    "status": {"code": 200},
                    "data": {"failed_uploads": [{"name": "train.jsonl"}]},
                },
            ),
        ):
            with pytest.raises(OutputError) as exc_info:
                await to_bailian_data([FileSpec(path=str(data_file))])
        assert exc_info.value.error_code == 4033

    @pytest.mark.asyncio
    async def test_unexpected_error_is_wrapped_as_4034(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"prompt": "hi"}', encoding="utf-8")

        with patch(
            "dashscope.applications.finetune.reinforcement.common.utils"
            ".async_http_request",
            new=AsyncMock(side_effect=ValueError("boom")),
        ):
            with pytest.raises(OutputError) as exc_info:
                await to_bailian_data([FileSpec(path=str(data_file))])
        assert exc_info.value.error_code == 4034
