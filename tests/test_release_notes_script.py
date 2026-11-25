# -*- coding: utf-8 -*-
"""Tests for tools/render_release_notes.py (draft Release body renderer)."""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "tools" / "render_release_notes.py"

spec = importlib.util.spec_from_file_location(
    "render_release_notes",
    SCRIPT_PATH,
)
rrn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrn)


def test_group_commits_by_conventional_type():
    messages = [
        "feat(cli): add expert command",
        "fix(sse): stop at DONE marker",
        "refactor(tokenizer): drop bundled vocab",
        "docs(en): refresh quickstart",
        "chore(release): bump version",
        "ci: pin cryptography",
        "test(cli): cover dumb terminal",
        "perf(http): reuse connections",
    ]
    groups = rrn.group_commits(messages)
    assert groups["FEATURES"] == "- **cli:** add expert command"
    assert groups["BUG_FIXES"] == "- **sse:** stop at DONE marker"
    assert groups["REFACTORING"] == "- **tokenizer:** drop bundled vocab"
    assert groups["DOCUMENTATION"] == "- **en:** refresh quickstart"
    assert groups["CI_CHORE"].splitlines() == [
        "- **release:** bump version",
        "- pin cryptography",
        "- **cli:** cover dumb terminal",
        "- **http:** reuse connections",
    ]
    assert groups["OTHER"] == "None"
    assert groups["BREAKING_CHANGES"] == "None"


def test_unscoped_and_untyped_subjects():
    groups = rrn.group_commits(["feat: bare feature", "random subject"])
    assert groups["FEATURES"] == "- bare feature"
    assert groups["OTHER"] == "- random subject"


def test_breaking_change_detection():
    messages = [
        "feat(api)!: rename chat parameter",
        "fix(core): adjust retry\n\nBREAKING CHANGE: retry default changed",
    ]
    groups = rrn.group_commits(messages)
    assert groups["BREAKING_CHANGES"].splitlines() == [
        "- **api:** rename chat parameter",
        "- **core:** adjust retry",
    ]
    # breaking commits also stay in their normal section
    assert groups["FEATURES"] == "- **api:** rename chat parameter"
    assert groups["BUG_FIXES"] == "- **core:** adjust retry"


def test_duplicate_subjects_collapse():
    groups = rrn.group_commits(["fix(a): same thing", "fix(a): same thing"])
    assert groups["BUG_FIXES"] == "- **a:** same thing"


def test_empty_sections_render_none():
    groups = rrn.group_commits([])
    for name, _, _ in rrn._SECTIONS:
        assert groups[name] == "None"
    assert groups["BREAKING_CHANGES"] == "None"


def test_render_body_fills_template():
    groups = rrn.group_commits(
        ["feat(cli): add expert command", "loose subject"],
    )
    body = rrn.render_body("v9.9.9", "https://example.com/compare", groups)
    assert "## v9.9.9 Release Notes" in body
    assert "pip install --upgrade dashscope==9.9.9" in body
    assert "- **cli:** add expert command" in body
    assert "### 📌 Other\n\n- loose subject" in body
    assert "https://example.com/compare" in body
    # no unsubstituted placeholders survive
    assert "${" not in body


def test_render_body_omits_empty_other_section():
    groups = rrn.group_commits(["fix(a): one"])
    body = rrn.render_body("v1.0.0", "https://example.com", groups)
    assert "### 📌 Other" not in body
    assert "### ⚠️ Breaking Changes\n\nNone" in body


def test_previous_tag_skips_current(monkeypatch):
    monkeypatch.setattr(
        rrn,
        "git",
        lambda *a: "v2.1.2\nv2.1.1\nv2.1.0\n",
    )
    assert rrn.previous_tag("v2.1.2") == "v2.1.1"
    assert rrn.previous_tag("v9.9.9") == "v2.1.2"


def test_previous_tag_first_release(monkeypatch):
    monkeypatch.setattr(rrn, "git", lambda *a: "v1.0.0\n")
    assert rrn.previous_tag("v1.0.0") == ""


def test_commit_messages_splits_on_nul(monkeypatch):
    captured = {}

    def fake_git(*args):
        captured["args"] = args
        return "feat: a\n\x00fix: b\n\nbody\x00"

    monkeypatch.setattr(rrn, "git", fake_git)
    messages = rrn.commit_messages("v1..v2")
    assert messages == ["feat: a", "fix: b\n\nbody"]
    assert captured["args"] == (
        "log",
        "--no-merges",
        "--format=%B%x00",
        "v1..v2",
    )


def test_main_writes_body(tmp_path, monkeypatch):
    monkeypatch.setattr(rrn, "previous_tag", lambda tag: "v1.0.0")
    monkeypatch.setattr(
        rrn,
        "commit_messages",
        lambda rev: ["feat(cli): shiny new thing"],
    )
    out = tmp_path / "release-body.md"
    rc = rrn.main(
        [
            "--tag",
            "v1.1.0",
            "--repo",
            "dashscope/dashscope-sdk-python",
            "--output",
            str(out),
        ],
    )
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert "- **cli:** shiny new thing" in body
    assert (
        "https://github.com/dashscope/dashscope-sdk-python"
        "/compare/v1.0.0...v1.1.0" in body
    )


@pytest.mark.parametrize("subject", ["Merge pull request #1", ""])
def test_non_conventional_never_crashes(subject):
    groups = rrn.group_commits([subject] if subject else [])
    assert isinstance(groups["OTHER"], str)
