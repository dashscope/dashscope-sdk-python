# -*- coding: utf-8 -*-
"""Test for base_address validation with DashScopeException."""

import pytest
from dashscope.common.error import DashScopeException
from dashscope.api_entities.api_request_factory import _build_api_request


class TestBaseAddressValidation:
    """Test cases for base_address validation using DashScopeException."""

    def test_http_base_address_without_scheme(self):
        """Test HTTP base_address without scheme raises DashScopeException."""
        with pytest.raises(DashScopeException, match=r"No scheme supplied"):
            _build_api_request(
                model="text-embedding-v2",
                input="hello",
                task_group="embeddings",
                task="text-embedding",
                function="text-embedding",
                api_key="test-api-key",
                base_address="POC_URL",
            )

    def test_https_base_address_with_scheme_should_work(self):
        """Test HTTPS base_address with scheme should not raise exception."""
        try:
            request = _build_api_request(
                model="text-embedding-v2",
                input="hello",
                task_group="embeddings",
                task="text-embedding",
                function="text-embedding",
                api_key="test-api-key",
                base_address="https://invalid.example.com",
            )
            # If we get here, URL validation passed
            assert request is not None
        except DashScopeException as e:
            if "No scheme supplied" in str(e):
                pytest.fail(
                    "Should not raise DashScopeException for valid https URL",
                )
            # Other DashScopeException are OK (e.g., network errors)

    def test_multimodal_embedding_invalid_base_address(self):
        """Test multimodal embedding with invalid base_address."""
        with pytest.raises(DashScopeException, match=r"No scheme supplied"):
            _build_api_request(
                model="multimodal-embedding-v1",
                input={"image": "test.jpg"},
                task_group="embeddings",
                task="multimodal-embedding",
                function="multimodal-embedding",
                api_key="test-api-key",
                base_address="INVALID_URL",
            )

    def test_websocket_invalid_base_address(self):
        """Test websocket with invalid base_address."""
        from dashscope.common.constants import ApiProtocol

        with pytest.raises(DashScopeException, match=r"No scheme supplied"):
            _build_api_request(
                model="tingwu-realtime",
                input={"audio": "test.wav"},
                task_group="",
                task="",
                function="",
                api_key="test-api-key",
                api_protocol=ApiProtocol.WEBSOCKET,
                base_address="INVALID_WS_URL",
            )

    def test_valid_websocket_base_address(self):
        """Test websocket with valid wss:// base_address."""
        from dashscope.common.constants import ApiProtocol

        try:
            request = _build_api_request(
                model="tingwu-realtime",
                input={"audio": "test.wav"},
                task_group="",
                task="",
                function="",
                api_key="test-api-key",
                api_protocol=ApiProtocol.WEBSOCKET,
                base_address="wss://valid.example.com",
            )
            assert request is not None
        except DashScopeException as e:
            if "No scheme supplied" in str(e):
                pytest.fail(
                    "Should not raise DashScopeException for valid wss:// URL",
                )
