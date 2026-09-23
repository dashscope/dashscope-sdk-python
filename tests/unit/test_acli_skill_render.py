# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Every bundled dashscope-sdk-expert skill must render through ``use_skill``.

This is the dashscope-only half of the ``render_skill`` regression net: the 13
skill files under ``examples/dashscope-sdk-expert/.acli/skills/`` ship in this
wheel and exist nowhere else, so they are the real corpus that broke.

``render_skill`` used to run ``str.format()`` over the whole skill body. A body
is Markdown full of literal JSON and Python braces, so ``text-generation.md`` --
which documents ``messages=[{"role": "user", ...}]`` and declares no arguments
at all -- raised ``KeyError('"role"')``. The old ``except KeyError`` turned that
into ``Error: missing args`` for a skill that takes none, so 6 of the 13 skills
were unusable through ``use_skill`` while ``/skill`` and the catalog kept
advertising them.

The assertions here are deliberately about observable output rather than about
the substitution mechanism: declared placeholders disappear, their values
arrive, and every other brace comes back byte-identical.
"""

# pylint: disable=protected-access

from __future__ import annotations

import re
from pathlib import Path

import pytest

import dashscope.acli
from dashscope.acli.skills.base import Skill, _parse_skill_md, render_skill

SKILLS_DIR = (
    Path(dashscope.acli.__file__).parent
    / "examples"
    / "dashscope-sdk-expert"
    / ".acli"
    / "skills"
)

# Distinct per position so a swapped argument order shows up as a missing
# sentinel instead of quietly passing.
SENTINEL = "ZZSENTINEL{}ZZ"

_BRACE_SPAN = re.compile(r"\{[^{}]*\}")


def _bundled_skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*.md"))


def _load(path: Path) -> Skill:
    skill = _parse_skill_md(path)
    assert skill is not None, f"{path.name} did not parse"
    return skill


def _args_for(skill: Skill) -> list[str]:
    return [SENTINEL.format(i) for i in range(len(skill.arguments))]


@pytest.fixture(scope="module")
def bundled_skills() -> list[Skill]:
    paths = _bundled_skill_files()
    assert paths, f"no skill files found under {SKILLS_DIR}"
    return [_load(p) for p in paths]


def test_the_bundled_corpus_is_the_one_that_ships(bundled_skills):
    """Guard against the test going vacuous if ``examples/`` is ever dropped.

    A silent sync that stops copying the skills directory would otherwise leave
    every assertion below passing over an empty corpus.
    """
    names = {s.name for s in bundled_skills}
    assert len(bundled_skills) == 13
    assert names == {path.stem for path in _bundled_skill_files()}
    # Both shapes matter: the bug hit no-argument skills carrying JSON hardest,
    # while the substitution path is only exercised by the ones with arguments.
    assert any(s.arguments for s in bundled_skills)
    assert any(not s.arguments for s in bundled_skills)
    assert sum(1 for s in bundled_skills if "{" in s.prompt_template) >= 10


@pytest.mark.parametrize("path", _bundled_skill_files(), ids=lambda p: p.stem)
def test_every_skill_renders_without_raising(path):
    skill = _load(path)
    rendered = render_skill(skill, _args_for(skill))
    assert rendered is not None, f"{skill.name} reported missing args"
    assert rendered.strip()


@pytest.mark.parametrize("path", _bundled_skill_files(), ids=lambda p: p.stem)
def test_undeclared_braces_are_left_exactly_as_authored(path):
    """A JSON sample in the body is documentation, not a placeholder."""
    skill = _load(path)
    rendered = render_skill(skill, _args_for(skill))
    assert rendered is not None

    declared = {f"{{{a}}}" for a in skill.arguments}
    literals = [
        span
        for span in _BRACE_SPAN.findall(skill.prompt_template)
        if span not in declared
    ]
    for span in literals:
        assert span in rendered, f"{skill.name} lost {span!r}"


@pytest.mark.parametrize("path", _bundled_skill_files(), ids=lambda p: p.stem)
def test_declared_arguments_are_substituted(path):
    skill = _load(path)
    args = _args_for(skill)
    rendered = render_skill(skill, args)
    assert rendered is not None

    for name, value in zip(skill.arguments, args):
        assert f"{{{name}}}" not in rendered, f"{skill.name} kept {{{name}}}"
        assert value in rendered, f"{skill.name} dropped the value of {name}"


def test_a_no_argument_skill_renders_to_its_own_body(bundled_skills):
    """The exact regression: ``text-generation`` takes no args and shows JSON."""
    skill = next(s for s in bundled_skills if s.name == "text-generation")
    assert skill.arguments == []
    assert '{"role": "user", "content": "Hello"}' in skill.prompt_template
    assert render_skill(skill, []) == skill.prompt_template


def test_too_few_arguments_still_reports_none(bundled_skills):
    """``None`` is the missing-args signal; it must stay reachable.

    Both ``use_skill`` and the scheduler branch on ``rendered is None`` to print
    the ``requires: <arg>`` hint. If rendering always succeeded, that hint would
    become dead code and a genuinely short call would send a half-substituted
    prompt to the model.
    """
    translate = next(s for s in bundled_skills if s.name == "translate")
    assert translate.arguments == ["target_lang", "text"]
    assert render_skill(translate, []) is None
    assert render_skill(translate, ["English"]) is None
    assert render_skill(translate, ["English", "hello there"]) is not None
