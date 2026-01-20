# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import Deployments
from tests.constants import TEST_JOB_ID
from tests.mock_request_base import MockRequestBase


class TestDeploymentRequest(MockRequestBase):
    def test_create_deployment_tune_job(self, http_server):
        resp = Deployments.call(
            model="gpt",
            suffix="1",
            capacity=2,
            headers={"X-Request-Id": "111111"},
        )
        assert resp.status_code == HTTPStatus.OK
        assert resp.output["deployed_model"] == "deploy123456"
        assert resp.output["status"] == "PENDING"

    def test_list_deployment_job(self, http_server):
        rsp = Deployments.list()
        assert rsp.status_code == HTTPStatus.OK
        assert len(rsp.output["deployments"]) == 1

    def test_get_deployment_job(self, http_server):
        rsp = Deployments.get(TEST_JOB_ID)
        assert rsp.status_code == HTTPStatus.OK
        assert rsp.output["deployed_model"] == TEST_JOB_ID
        assert rsp.output["status"] == "PENDING"

    def test_delete_deployment_job(self, http_server):
        rsp = Deployments.delete(TEST_JOB_ID)
        assert rsp.status_code == HTTPStatus.OK
