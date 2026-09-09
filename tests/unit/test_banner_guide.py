# -*- coding: utf-8 -*-
"""The startup banner surfaces the embedded host's guide link."""

from dashscope.acli.cli.startup import _print_banner
from dashscope.acli.config import Config

GUIDE = "https://help.aliyun.com/en/model-studio/dashscope-sdk-expert"


def test_banner_shows_guide_url(capsys):
    config = Config()
    config._embedded_guide_url = GUIDE
    _print_banner(config)
    out = capsys.readouterr().out
    assert "Guide:" in out
    assert GUIDE in out


def test_banner_omits_guide_url_by_default(capsys):
    _print_banner(Config())
    assert "Guide:" not in capsys.readouterr().out


def test_embedded_run_stores_guide_url(monkeypatch):
    import dashscope.acli.ui.embedded as embedded

    captured = {}
    monkeypatch.setattr(
        Config,
        "load",
        classmethod(lambda cls, **kw: Config()),
    )

    async def fake_oneshot(config, prompt, system_prompt):
        captured["config"] = config

    monkeypatch.setattr(embedded, "_run_oneshot_embedded", fake_oneshot)
    embedded.run(command="hi", guide_url=GUIDE)
    assert captured["config"]._embedded_guide_url == GUIDE


def test_embedded_run_defaults_to_no_guide_url(monkeypatch):
    import dashscope.acli.ui.embedded as embedded

    captured = {}
    monkeypatch.setattr(
        Config,
        "load",
        classmethod(lambda cls, **kw: Config()),
    )

    async def fake_oneshot(config, prompt, system_prompt):
        captured["config"] = config

    monkeypatch.setattr(embedded, "_run_oneshot_embedded", fake_oneshot)
    embedded.run(command="hi")
    assert getattr(captured["config"], "_embedded_guide_url", "") == ""
