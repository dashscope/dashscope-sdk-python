# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import Deployments
from tests.unit.constants import TEST_JOB_ID
from tests.unit.mock_request_base import MockRequestBase


class TestDeploymentRequest(MockRequestBase):
    # pylint: disable=unused-argument
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

    def test_create_deployment_with_plan(self, http_server):
        # Regression: plan/template_id must reach the request body (verified
        # via mock echo; previously dropped silently, causing "plan is blank")
        resp = Deployments.call(
            model="gpt",
            suffix="1",
            capacity=2,
            plan="plan-standard",
            template_id="tpl-001",
        )
        assert resp.status_code == HTTPStatus.OK
        assert resp.output["deployed_model"] == "deploy123456"
        assert resp.output["plan"] == "plan-standard"
        assert resp.output["template_id"] == "tpl-001"

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
