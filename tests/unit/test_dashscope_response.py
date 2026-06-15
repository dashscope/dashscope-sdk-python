# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from dashscope.api_entities.dashscope_response import DictMixin


class TestDictMixin:
    def test_getattr_missing_key_raises_attribute_error(self):
        response = DictMixin(existing="value")

        try:
            response.missing
        except AttributeError:
            return

        raise AssertionError("Missing attribute should raise AttributeError")

    def test_getattr_existing_key_returns_value(self):
        response = DictMixin(existing="value")

        assert response.existing == "value"
