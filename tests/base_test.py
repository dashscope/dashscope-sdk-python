# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import dashscope


class BaseTestEnvironment:
    @classmethod
    def setup_class(cls):
        cls.origin_api_key = dashscope.api_key
        cls.origin_base_http_api_url = dashscope.base_http_api_url
        cls.origin_base_ws_api_url = dashscope.base_websocket_api_url

    @classmethod
    def teardown_class(cls):
        dashscope.api_key = cls.origin_api_key
        dashscope.base_http_api_url = cls.origin_base_http_api_url
        dashscope.base_websocket_api_url = cls.origin_base_ws_api_url
