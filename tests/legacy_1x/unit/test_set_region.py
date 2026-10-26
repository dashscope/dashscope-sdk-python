# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for dashscope.set_region() function."""

import pytest

import dashscope


class TestSetRegion:
    """Test suite for set_region() function."""

    def setup_method(self):
        """Reset to default URLs before each test."""
        # Store original values
        self.original_http = dashscope.base_http_api_url
        self.original_compatible = dashscope.base_compatible_api_url
        self.original_websocket = dashscope.base_websocket_api_url

    def teardown_method(self):
        """Restore original URLs after each test."""
        dashscope.base_http_api_url = self.original_http
        dashscope.base_compatible_api_url = self.original_compatible
        dashscope.base_websocket_api_url = self.original_websocket

    def test_set_region_ap_southeast_1(self):
        """Test switching to Singapore region."""
        dashscope.set_region("ap-southeast-1", "ws-test123")

        assert (
            dashscope.base_http_api_url
            == "https://ws-test123.ap-southeast-1.maas.aliyuncs.com/api/v1"
        )
        assert (
            dashscope.base_compatible_api_url
            == "https://ws-test123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_set_region_us_east_1(self):
        """Test switching to US East region."""
        dashscope.set_region("us-east-1", "ws-us-456")

        assert (
            dashscope.base_http_api_url
            == "https://ws-us-456.us-east-1.maas.aliyuncs.com/api/v1"
        )
        assert (
            dashscope.base_compatible_api_url
            == "https://ws-us-456.us-east-1.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_set_region_cn_hongkong(self):
        """Test switching to Hong Kong region."""
        dashscope.set_region("cn-hongkong", "ws-hk-789")

        assert (
            dashscope.base_http_api_url
            == "https://ws-hk-789.cn-hongkong.maas.aliyuncs.com/api/v1"
        )
        assert (
            dashscope.base_compatible_api_url
            == "https://ws-hk-789.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_set_region_cn_beijing(self):
        """Test switching to Beijing region."""
        dashscope.set_region("cn-beijing", "ws-bj-001")

        assert (
            dashscope.base_http_api_url
            == "https://ws-bj-001.cn-beijing.maas.aliyuncs.com/api/v1"
        )
        assert (
            dashscope.base_compatible_api_url
            == "https://ws-bj-001.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_set_region_eu_central_1(self):
        """Test switching to Europe region."""
        dashscope.set_region("eu-central-1", "ws-eu-123")

        assert (
            dashscope.base_http_api_url
            == "https://ws-eu-123.eu-central-1.maas.aliyuncs.com/api/v1"
        )
        assert (
            dashscope.base_compatible_api_url
            == "https://ws-eu-123.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_set_region_ap_northeast_1(self):
        """Test switching to Japan region."""
        dashscope.set_region("ap-northeast-1", "ws-jp-456")

        assert (
            dashscope.base_http_api_url
            == "https://ws-jp-456.ap-northeast-1.maas.aliyuncs.com/api/v1"
        )
        assert (
            dashscope.base_compatible_api_url
            == "https://ws-jp-456.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_set_region_updates_websocket_url(self):
        """Test that set_region also switches the websocket URL."""
        dashscope.set_region("ap-southeast-1", "ws-test123")

        assert dashscope.base_websocket_api_url == (
            "wss://ws-test123.ap-southeast-1.maas.aliyuncs.com"
            "/api-ws/v1/inference"
        )

    def test_set_region_invalid_region(self):
        """Test that invalid region raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dashscope.set_region("mars-1", "ws-xxx")

        assert "Unsupported region 'mars-1'" in str(exc_info.value)
        assert "Supported regions:" in str(exc_info.value)

    def test_set_region_missing_workspace_id(self):
        """Test that missing workspace_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dashscope.set_region("cn-beijing")

        assert "workspace_id is required" in str(exc_info.value)

    def test_set_region_empty_workspace_id(self):
        """Test that empty workspace_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dashscope.set_region("cn-beijing", "")

        assert "workspace_id is required" in str(exc_info.value)

    def test_set_region_none_workspace_id(self):
        """Test that None workspace_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dashscope.set_region("cn-beijing", None)

        assert "workspace_id is required" in str(exc_info.value)

    @pytest.mark.parametrize(
        "bad_workspace_id",
        [
            "evil.com/x",
            "evil.com?x=1",
            "evil.com#frag",
            "a@evil.com",
            "evil.com:443",
            "evil com",
            "evil.com",
            "-evil",
            "a" * 65,
        ],
    )
    def test_set_region_invalid_workspace_id(self, bad_workspace_id):
        """workspace_id with URL-breaking characters must be rejected."""
        with pytest.raises(ValueError) as exc_info:
            dashscope.set_region("ap-southeast-1", bad_workspace_id)

        assert "Invalid workspace_id" in str(exc_info.value)
        # Globals must not be modified on rejection.
        assert bad_workspace_id not in dashscope.base_http_api_url

    def test_set_region_multiple_times(self):
        """Test switching between multiple regions."""
        # First region
        dashscope.set_region("ap-southeast-1", "ws-sg-111")
        assert "ap-southeast-1" in dashscope.base_http_api_url
        assert "ws-sg-111" in dashscope.base_http_api_url

        # Second region
        dashscope.set_region("us-east-1", "ws-us-222")
        assert "us-east-1" in dashscope.base_http_api_url
        assert "ws-us-222" in dashscope.base_http_api_url

        # Third region
        dashscope.set_region("cn-beijing", "ws-bj-333")
        assert "cn-beijing" in dashscope.base_http_api_url
        assert "ws-bj-333" in dashscope.base_http_api_url

    def test_set_region_url_format(self):
        """Test that URLs follow the correct format."""
        dashscope.set_region("ap-southeast-1", "ws-test-123")

        # Check HTTP API URL format
        assert dashscope.base_http_api_url.startswith("https://")
        assert dashscope.base_http_api_url.endswith("/api/v1")
        assert "ws-test-123.ap-southeast-1.maas.aliyuncs.com" in (
            dashscope.base_http_api_url
        )

        # Check Compatible API URL format
        assert dashscope.base_compatible_api_url.startswith("https://")
        assert dashscope.base_compatible_api_url.endswith(
            "/compatible-mode/v1",
        )
        assert "ws-test-123.ap-southeast-1.maas.aliyuncs.com" in (
            dashscope.base_compatible_api_url
        )

    def test_set_region_all_supported_regions(self):
        """Test that all documented regions work."""
        expected_regions = {
            "cn-beijing",
            "ap-southeast-1",
            "us-east-1",
            "cn-hongkong",
            "eu-central-1",
            "ap-northeast-1",
        }

        for region in expected_regions:
            dashscope.set_region(region, "ws-test")
            assert region in dashscope.base_http_api_url

    def test_set_region_exported(self):
        """Test that set_region is exported in __all__."""
        assert "set_region" in dashscope.__all__

    def test_set_region_does_not_affect_original_defaults(self):
        """Test that original defaults are preserved after reset."""
        original_http = "https://dashscope.aliyuncs.com/api/v1"
        original_compatible = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Set to defaults
        dashscope.base_http_api_url = original_http
        dashscope.base_compatible_api_url = original_compatible

        # Change region
        dashscope.set_region("ap-southeast-1", "ws-test")

        # Should have changed
        assert dashscope.base_http_api_url != original_http

        # Manually reset (simulating what teardown does)
        dashscope.base_http_api_url = original_http
        dashscope.base_compatible_api_url = original_compatible

        # Should be back to original
        assert dashscope.base_http_api_url == original_http
        assert dashscope.base_compatible_api_url == original_compatible
