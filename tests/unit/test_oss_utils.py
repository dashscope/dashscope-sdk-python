# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

import pytest

from dashscope.common.error import InvalidInput
from dashscope.utils import oss_utils
from dashscope.utils.oss_utils import OssUtils


class FakeUploadResponse:
    status_code = HTTPStatus.OK
    headers = {}


class FakeSession:
    captured_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def post(self, url, files, data, headers, timeout):
        assert url == "https://oss.example.com"
        assert data["key"] == "test-dir/dogs.jpg"
        assert headers["Accept"] == "application/json"
        assert timeout == 3600

        FakeSession.captured_file = files["file"]
        assert not FakeSession.captured_file.closed
        return FakeUploadResponse()


class TestOssUtils:
    def test_upload_closes_opened_file(self, monkeypatch):
        upload_certificate = {
            "oss_access_key_id": "access-key-id",
            "signature": "signature",
            "policy": "policy",
            "upload_dir": "test-dir",
            "x_oss_object_acl": "private",
            "x_oss_forbid_overwrite": "true",
            "upload_host": "https://oss.example.com",
        }
        FakeSession.captured_file = None
        monkeypatch.setattr(oss_utils.requests, "Session", FakeSession)

        file_url, returned_certificate = OssUtils.upload(
            model="test-model",
            file_path="tests/data/dogs.jpg",
            api_key="test-api-key",
            upload_certificate=upload_certificate,
        )

        assert file_url == "oss://test-dir/dogs.jpg"
        assert returned_certificate is upload_certificate
        assert FakeSession.captured_file is not None
        assert FakeSession.captured_file.closed

    def test_check_and_upload_local_uploads_relative_file_uri(
        self,
        monkeypatch,
    ):
        captured_file_path = {}

        def fake_isfile(file_path):
            captured_file_path["value"] = file_path
            return True

        def fake_upload(model, file_path, api_key, upload_certificate):
            assert model == "test-model"
            assert api_key == "test-api-key"
            assert upload_certificate == {"cert": "value"}
            assert file_path == "test_video_frames/frame_0000.jpg"
            return "oss://test-dir/frame_0000.jpg", {"cert": "value"}

        monkeypatch.setattr(oss_utils.os.path, "isfile", fake_isfile)
        monkeypatch.setattr(OssUtils, "upload", fake_upload)

        is_upload, file_url, certificate = oss_utils.check_and_upload_local(
            model="test-model",
            content="file://test_video_frames/frame_0000.jpg",
            api_key="test-api-key",
            upload_certificate={"cert": "value"},
        )

        assert is_upload
        assert file_url == "oss://test-dir/frame_0000.jpg"
        assert certificate == {"cert": "value"}
        assert (
            captured_file_path["value"] == "test_video_frames/frame_0000.jpg"
        )

    def test_check_and_upload_local_supports_windows_absolute_file_uri(
        self,
        monkeypatch,
    ):
        captured_file_path = {}

        def fake_isfile(file_path):
            captured_file_path["value"] = file_path
            return True

        def fake_upload(
            model,
            file_path,
            api_key,
            upload_certificate,
        ):
            assert model == "test-model"
            assert file_path == "C:/Users/test/frame_0000.jpg"
            assert api_key == "test-api-key"
            return "oss://test-dir/frame_0000.jpg", upload_certificate

        monkeypatch.setattr(oss_utils.os.path, "isfile", fake_isfile)
        monkeypatch.setattr(OssUtils, "upload", fake_upload)

        is_upload, file_url, _ = oss_utils.check_and_upload_local(
            model="test-model",
            content="file:///C:/Users/test/frame_0000.jpg",
            api_key="test-api-key",
        )

        assert is_upload
        assert file_url == "oss://test-dir/frame_0000.jpg"
        assert captured_file_path["value"] == "C:/Users/test/frame_0000.jpg"

    def test_check_and_upload_local_raises_when_file_uri_not_found(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            oss_utils.os.path,
            "isfile",
            lambda file_path: False,
        )

        with pytest.raises(InvalidInput):
            oss_utils.check_and_upload_local(
                model="test-model",
                content="file://missing/frame_0000.jpg",
                api_key="test-api-key",
            )
