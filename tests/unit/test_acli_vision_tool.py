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


class CountingProvider:
    def __init__(self):
        self.calls = 0
        self.questions = []

    async def chat(self, messages):
        self.calls += 1
        self.questions.append(messages[0]["content"][0]["text"])
        return SimpleNamespace(content=f"slice text {self.calls}")


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

    async def test_tall_image_auto_sliced(self, tmp_path):
        PIL = pytest.importorskip("PIL")
        provider = CountingProvider()
        _register(provider)

        img = PIL.Image.new("RGB", (200, 10000), "white")
        path = tmp_path / "tall.png"
        img.save(path)

        result = await registry.get("view_image").func(
            image_path=str(path),
        )

        expected = len(vision.plan_slices(200, 10000))
        assert provider.calls == expected
        assert all(q.startswith("Slice ") for q in provider.questions)
        assert result.startswith("(Image auto-split into")
        assert f"--- slice 1/{expected} ---" in result
        assert f"--- slice {expected}/{expected} ---" in result

    async def test_slice_never_sends_whole_image(self, tmp_path):
        PIL = pytest.importorskip("PIL")
        provider = CountingProvider()
        _register(provider)

        img = PIL.Image.new("RGB", (200, 10000), "white")
        path = tmp_path / "tall.png"
        img.save(path)

        result = await registry.get("view_image").func(
            image_path=str(path),
            slice_mode="never",
        )

        assert provider.calls == 1
        assert result == "slice text 1"

    async def test_slice_error_reported_per_slice(self, tmp_path):
        PIL = pytest.importorskip("PIL")

        class BrokenProvider:
            async def chat(self, messages):
                raise RuntimeError("boom")

        _register(BrokenProvider())
        img = PIL.Image.new("RGB", (200, 10000), "white")
        path = tmp_path / "tall.png"
        img.save(path)

        result = await registry.get("view_image").func(
            image_path=str(path),
        )
        assert "Error: boom" in result
        assert "--- slice 1/" in result


class TestPlanSlices:
    def test_normal_image_returns_none(self):
        assert vision.plan_slices(1920, 1080) is None
        assert vision.plan_slices(8000, 2000) is None

    def test_tall_image_slices_vertically(self):
        boxes = vision.plan_slices(1382, 12028)
        assert boxes is not None
        assert all(b[0] == 0 and b[2] == 1382 for b in boxes)
        assert boxes[0][1] == 0
        assert boxes[-1][3] == 12028
        for prev, nxt in zip(boxes, boxes[1:]):
            assert nxt[1] < prev[3]  # overlap
            assert nxt[1] == prev[3] - vision._SLICE_OVERLAP

    def test_wide_image_slices_horizontally(self):
        boxes = vision.plan_slices(20000, 800)
        assert boxes is not None
        assert all(b[1] == 0 and b[3] == 800 for b in boxes)
        assert boxes[-1][2] == 20000

    def test_extreme_image_capped(self):
        boxes = vision.plan_slices(1000, 1000000)
        assert len(boxes) <= vision._SLICE_MAX_COUNT
        assert boxes[-1][3] == 1000000

    def test_high_ratio_triggers_even_when_short(self):
        assert vision.plan_slices(1000, 4500) is not None
        assert vision.plan_slices(4500, 1000) is not None


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
