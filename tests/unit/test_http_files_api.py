# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import Files
from tests.unit.mock_request_base import MockRequestBase


class TestFileRequest(MockRequestBase):
    # pylint: disable=unused-argument
    def test_upload_files(self, http_server):
        resp = Files.upload(
            file_path="tests/data/dogs.jpg",
            purpose="fine-tune",
            custom_file_name="gpt3_training.csv",
        )
        print(resp)
        assert resp.status_code == HTTPStatus.OK
        assert resp.output["uploaded_files"][0]["file_id"] == "xxxx"

    def test_list_files(self, http_server):
        resp = Files.list()
        assert resp.status_code == HTTPStatus.OK
        assert len(resp.output["files"]) == 3

    def test_get_file(self, http_server):
        resp = Files.get(file_id="111111")
        assert resp.status_code == HTTPStatus.OK
        assert resp.output["file_id"] == "111111"

    def test_delete_file(self, http_server):
        resp = Files.delete(file_id="111111")
        assert resp.status_code == HTTPStatus.OK

        resp = Files.delete(file_id="222222")  # not exist
        assert resp.status_code == HTTPStatus.NOT_FOUND
        resp = Files.delete(file_id="333333")  # no permission
        assert resp.status_code == HTTPStatus.FORBIDDEN
        resp = Files.delete(file_id="444444", api_key="api-key")  # not exist
        assert resp.status_code == HTTPStatus.UNAUTHORIZED
