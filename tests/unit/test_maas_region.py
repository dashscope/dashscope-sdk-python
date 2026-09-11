# -*- coding: utf-8 -*-
"""Tests for MaaS international region URL resolution."""
import os
import unittest
from unittest.mock import patch

from dashscope.common.env import MAAS_REGIONS, resolve_base_url
from dashscope.common.error import InputRequired


class TestResolveBaseUrl(unittest.TestCase):
    """Test resolve_base_url helper."""

    def test_no_placeholder(self):
        url = "https://dashscope.aliyuncs.com/api/v1"
        self.assertEqual(resolve_base_url(url, "ws-123"), url)

    def test_resolve_with_workspace(self):
        url = "https://{workspace_id}.ap-southeast-1.maas.aliyuncs.com/api/v1"
        result = resolve_base_url(url, "ws-abc")
        self.assertEqual(
            result,
            "https://ws-abc.ap-southeast-1.maas.aliyuncs.com/api/v1",
        )

    def test_resolve_without_workspace_raises(self):
        import dashscope.common.env as env_mod

        url = "https://{workspace_id}.us-east-1.maas.aliyuncs.com/api/v1"
        with patch.object(env_mod, "_default_workspace_id", None):
            with self.assertRaises(InputRequired):
                resolve_base_url(url, None)

    def test_resolve_with_env_default_workspace(self):
        import dashscope.common.env as env_mod

        url = "https://{workspace_id}.us-east-1.maas.aliyuncs.com/api/v1"
        with patch.object(env_mod, "_default_workspace_id", "ws-env"):
            result = resolve_base_url(url, None)
        self.assertEqual(
            result,
            "https://ws-env.us-east-1.maas.aliyuncs.com/api/v1",
        )

    def test_resolve_all_regions(self):
        for region_id in MAAS_REGIONS.values():
            url = f"https://{{workspace_id}}.{region_id}.maas.aliyuncs.com/api/v1"
            result = resolve_base_url(url, "my-ws")
            self.assertEqual(
                result,
                f"https://my-ws.{region_id}.maas.aliyuncs.com/api/v1",
            )


class TestMaasRegions(unittest.TestCase):
    """Test MAAS_REGIONS constant."""

    def test_expected_regions(self):
        expected = {
            "ap-southeast-1",
            "us-east-1",
            "cn-hongkong",
            "eu-central-1",
            "ap-northeast-1",
        }
        self.assertEqual(set(MAAS_REGIONS.keys()), expected)


class TestMaasEnvLoading(unittest.TestCase):
    """Test that env.py loads correct URLs for MaaS regions."""

    def _reload_env(self, env_vars):
        """Reload env module with given env vars."""
        import importlib
        import dashscope.common.env as env_mod

        with patch.dict(os.environ, env_vars, clear=False):
            importlib.reload(env_mod)
        return env_mod

    def _restore_env(self):
        """Restore env module to default state."""
        import importlib
        import dashscope.common.env as env_mod

        with patch.dict(
            os.environ,
            {"DASHSCOPE_API_REGION": "cn-beijing"},
            clear=False,
        ):
            # Remove MaaS-related vars if present
            for key in [
                "DASHSCOPE_WORKSPACE_ID",
                "DASHSCOPE_HTTP_BASE_URL",
                "DASHSCOPE_WEBSOCKET_BASE_URL",
                "DASHSCOPE_COMPATIBLE_BASE_URL",
            ]:
                os.environ.pop(key, None)
            importlib.reload(env_mod)

    def tearDown(self):
        self._restore_env()

    def test_default_region_legacy_urls(self):
        env = self._reload_env({"DASHSCOPE_API_REGION": "cn-beijing"})
        self.assertEqual(
            env.base_http_api_url,
            "https://dashscope.aliyuncs.com/api/v1",
        )
        self.assertEqual(
            env.base_compatible_api_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_maas_region_with_workspace(self):
        env = self._reload_env(
            {
                "DASHSCOPE_API_REGION": "ap-southeast-1",
                "DASHSCOPE_WORKSPACE_ID": "ws-test-123",
            },
        )
        self.assertEqual(
            env.base_http_api_url,
            "https://ws-test-123.ap-southeast-1.maas.aliyuncs.com/api/v1",
        )
        self.assertEqual(
            env.base_compatible_api_url,
            "https://ws-test-123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )

    def test_maas_region_without_workspace(self):
        env = self._reload_env(
            {"DASHSCOPE_API_REGION": "us-east-1"},
        )
        self.assertIn("{workspace_id}", env.base_http_api_url)
        self.assertIn("us-east-1.maas.aliyuncs.com", env.base_http_api_url)

    def test_maas_region_resolve_at_call_time(self):
        """workspace passed at call time resolves the placeholder."""
        env = self._reload_env(
            {"DASHSCOPE_API_REGION": "eu-central-1"},
        )
        resolved = env.resolve_base_url(
            env.base_http_api_url,
            workspace_id="ws-runtime",
        )
        self.assertEqual(
            resolved,
            "https://ws-runtime.eu-central-1.maas.aliyuncs.com/api/v1",
        )

    def test_env_override_takes_precedence(self):
        env = self._reload_env(
            {
                "DASHSCOPE_API_REGION": "ap-southeast-1",
                "DASHSCOPE_HTTP_BASE_URL": "https://custom.host/api/v1",
            },
        )
        self.assertEqual(
            env.base_http_api_url,
            "https://custom.host/api/v1",
        )


class TestMaasCallTimeResolution(unittest.TestCase):
    """Per-call workspace must resolve the {workspace_id} placeholder
    on the main request paths.
    """

    PLACEHOLDER_URL = (
        "https://{workspace_id}.ap-southeast-1.maas.aliyuncs.com/api/v1"
    )

    def setUp(self):
        import dashscope

        self._orig_http_url = dashscope.base_http_api_url
        dashscope.base_http_api_url = self.PLACEHOLDER_URL

    def tearDown(self):
        import dashscope

        dashscope.base_http_api_url = self._orig_http_url

    def _capture_request_url(self, func):
        from dashscope.client import base_api as base_api_mod

        class ShortCircuit(Exception):
            pass

        captured = {}
        orig_build = (
            base_api_mod._build_api_request  # pylint: disable=protected-access
        )

        def spy(*args, **kwargs):
            request = orig_build(*args, **kwargs)
            captured["url"] = request.url
            raise ShortCircuit()

        with patch.object(base_api_mod, "_build_api_request", spy):
            with self.assertRaises(ShortCircuit):
                func()
        return captured["url"]

    def test_generation_call_resolves_workspace(self):
        from dashscope import Generation

        url = self._capture_request_url(
            lambda: Generation.call(
                model="qwen-turbo",
                prompt="hi",
                workspace="ws-xyz",
                api_key="sk-test",
            ),
        )
        self.assertEqual(
            url,
            "https://ws-xyz.ap-southeast-1.maas.aliyuncs.com/api/v1"
            "/services/aigc/text-generation/generation",
        )

    def test_async_task_call_resolves_workspace(self):
        from dashscope import ImageSynthesis

        url = self._capture_request_url(
            lambda: ImageSynthesis.call(
                model="wanx-v1",
                prompt="hi",
                workspace="ws-xyz",
                api_key="sk-test",
            ),
        )
        self.assertEqual(
            url,
            "https://ws-xyz.ap-southeast-1.maas.aliyuncs.com/api/v1"
            "/services/aigc/text2image/image-synthesis",
        )

    def test_call_without_workspace_raises_clear_error(self):
        import dashscope.common.env as env_mod
        from dashscope import Generation

        with patch.object(env_mod, "_default_workspace_id", None):
            with self.assertRaises(InputRequired) as ctx:
                Generation.call(
                    model="qwen-turbo",
                    prompt="hi",
                    api_key="sk-test",
                )
        self.assertIn("workspace", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
