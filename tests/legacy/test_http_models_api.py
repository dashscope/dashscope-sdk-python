# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import Models
from tests.unit.constants import TEST_JOB_ID
from tests.unit.mock_request_base import MockRequestBase


class TestModelRequest(MockRequestBase):
    def test_list_models(self, http_server):
        rsp = Models.list()
        assert rsp.status_code == HTTPStatus.OK
        assert len(rsp.output["models"]) == 2

    def test_get_model(self, http_server):
        rsp = Models.get(TEST_JOB_ID)
        assert rsp.status_code == HTTPStatus.OK
        assert rsp.output["model_id"] == TEST_JOB_ID
