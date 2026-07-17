# -*- coding: utf-8 -*-
"""Test for InvalidBaseURL exception handling."""

import pytest
from dashscope.common.error import InvalidBaseURL
from dashscope.api_entities.api_request_factory import _build_api_request


class TestInvalidBaseURL:
    """Test cases for invalid base_address validation."""

    def test_http_base_address_without_scheme(self):
        """Test that HTTP base_address without scheme raises InvalidBaseURL."""
        with pytest.raises(InvalidBaseURL, match=r"No scheme supplied"):
            _build_api_request(
                model="text-embedding-v2",
                input="hello",
                task_group="embeddings",
                task="text-embedding",
                function="text-embedding",
                api_key="test-api-key",
                base_address="POC_URL"
            )

    def test_https_base_address_with_scheme_should_work(self):
        """Test that HTTPS base_address with scheme should not raise InvalidBaseURL at validation stage."""
        # This should pass URL validation (will fail later at network level)
        try:
            request = _build_api_request(
                model="text-embedding-v2",
                input="hello",
                task_group="embeddings",
                task="text-embedding",
                function="text-embedding",
                api_key="test-api-key",
                base_address="https://invalid.example.com"
            )
            # If we get here, URL validation passed
            assert request is not None
        except InvalidBaseURL:
            pytest.fail("Should not raise InvalidBaseURL for valid https URL")

    def test_multimodal_embedding_invalid_base_address(self):
        """Test multimodal embedding with invalid base_address."""
        with pytest.raises(InvalidBaseURL, match=r"No scheme supplied"):
            _build_api_request(
                model="multimodal-embedding-v1",
                input={"image": "test.jpg"},
                task_group="embeddings",
                task="multimodal-embedding",
                function="multimodal-embedding",
                api_key="test-api-key",
                base_address="INVALID_URL"
            )

    def test_websocket_invalid_base_address(self):
        """Test websocket with invalid base_address."""
        from dashscope.common.constants import ApiProtocol
        
        with pytest.raises(InvalidBaseURL, match=r"No scheme supplied"):
            _build_api_request(
                model="tingwu-realtime",
                input=None,
                task_group="",
                task="",
                function="",
                api_key="test-api-key",
                api_protocol=ApiProtocol.WEBSOCKET,
                base_address="INVALID_WS_URL"
            )

    def test_valid_websocket_base_address(self):
        """Test websocket with valid wss:// base_address."""
        from dashscope.common.constants import ApiProtocol
        
        try:
            request = _build_api_request(
                model="tingwu-realtime",
                input={"audio": "test.wav"},  # Provide input data
                task_group="",
                task="",
                function="",
                api_key="test-api-key",
                api_protocol=ApiProtocol.WEBSOCKET,
                base_address="wss://valid.example.com"
            )
            assert request is not None
        except InvalidBaseURL:
            pytest.fail("Should not raise InvalidBaseURL for valid wss:// URL")
