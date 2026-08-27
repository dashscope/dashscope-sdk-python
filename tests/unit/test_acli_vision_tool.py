# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# pylint: disable=protected-access

from types import SimpleNamespace

import pytest

from dashscope.acli.tools import vision
from dashscope.acli.tools.registry import registry


class FakeProvider:
    def __init__(self):
        self.messages = None

    async def chat(self, messages):
        self.messages = messages
        return SimpleNamespace(content="transcribed table text")


def _config():
    return SimpleNamespace(
        provider="tongyi",
        model="qwen3.8-max",
        api_key="sk-test",
        base_url="",
    )


def _register(fake_provider):
    captured = {}

    def get_provider_fn(provider, model, api_key, base_url=None):
        captured.update(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        return fake_provider

    vision.register_vision_tools(
        _config(),
        get_provider_fn=get_provider_fn,
    )
    return captured


class TestViewImage:
    async def test_sends_image_to_vision_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            vision,
            "is_vision_model",
            lambda model: False,
        )
        monkeypatch.setattr(
            "dashscope.acli.extensions.find_provider",
            lambda name: SimpleNamespace(vision_models=["qwen-vl-plus"]),
        )
        fake_provider = FakeProvider()
        captured = _register(fake_provider)

        image = tmp_path / "shot.png"
        image.write_bytes(b"\x89PNG fake bytes")

        result = await registry.get("view_image").func(
            image_path=str(image),
        )

        assert result == "transcribed table text"
        assert captured["model"] == "qwen-vl-plus"
        content = fake_provider.messages[0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,",
        )

    async def test_missing_file_returns_error(self):
        _register(FakeProvider())
        result = await registry.get("view_image").func(
            image_path=str("no-such-image.png"),
        )
        assert result.startswith("Error: failed to read image")

    async def test_vision_call_failure_returns_error(self, tmp_path):
        class BrokenProvider:
            async def chat(self, messages):
                raise RuntimeError("boom")

        _register(BrokenProvider())
        image = tmp_path / "shot.png"
        image.write_bytes(b"\x89PNG fake bytes")
        result = await registry.get("view_image").func(
            image_path=str(image),
        )
        assert result == "Error: vision call failed: boom"


class TestPickVisionModel:
    def test_active_model_when_vision_capable(self, monkeypatch):
        monkeypatch.setattr(vision, "is_vision_model", lambda m: True)
        assert vision.pick_vision_model(_config()) == "qwen3.8-max"

    def test_extension_vision_models(self, monkeypatch):
        monkeypatch.setattr(vision, "is_vision_model", lambda m: False)
        monkeypatch.setattr(
            "dashscope.acli.extensions.find_provider",
            lambda name: SimpleNamespace(vision_models=["qwen-vl-max"]),
        )
        assert vision.pick_vision_model(_config()) == "qwen-vl-max"

    def test_fallback_per_provider(self, monkeypatch):
        monkeypatch.setattr(vision, "is_vision_model", lambda m: False)
        monkeypatch.setattr(
            "dashscope.acli.extensions.find_provider",
            lambda name: None,
        )
        assert vision.pick_vision_model(_config()) == "qwen-vl-max"
        cfg = _config()
        cfg.provider = "openai"
        assert vision.pick_vision_model(cfg) == "gpt-4o"


class TestSchemaExposure:
    def test_view_image_always_offered(self):
        vision.register_vision_tools(
            _config(),
            get_provider_fn=lambda *a, **k: FakeProvider(),
        )
        names = [t["name"] for t in registry.to_schema_list()]
        assert "view_image" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
