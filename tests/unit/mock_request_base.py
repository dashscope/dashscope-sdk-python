# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import dashscope
from tests.unit.base_test import BaseTestEnvironment


class MockRequestBase(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        dashscope.base_http_api_url = "http://localhost:8080/api/v1/"
        dashscope.api_key = "default"
        dashscope.api_protocol = "http"


class MockServerBase(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        dashscope.base_http_api_url = "http://localhost:8089/api/v1/"
        dashscope.base_websocket_api_url = (
            "http://localhost:8089/api-ws/v1/inference"
        )
        dashscope.api_key = "default"
        dashscope.api_protocol = "http"
