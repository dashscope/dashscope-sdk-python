# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for issue #207.

The deprecated Assistants API must not emit a DeprecationWarning on a plain
``import dashscope``; it should warn only when a deprecated name is actually
accessed. These run in subprocesses so each starts from a clean import state.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(snippet, *, error_on_deprecation):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    args = [sys.executable]
    if error_on_deprecation:
        args += ["-W", "error::DeprecationWarning"]
    args += ["-c", textwrap.dedent(snippet)]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )


def test_plain_import_emits_no_deprecation_warning():
    result = _run("import dashscope", error_on_deprecation=True)
    assert result.returncode == 0, result.stderr


def test_non_assistants_usage_emits_no_deprecation_warning():
    result = _run(
        "import dashscope; dashscope.Generation; dashscope.TextEmbedding",
        error_on_deprecation=True,
    )
    assert result.returncode == 0, result.stderr


def test_accessing_assistants_emits_deprecation_warning():
    result = _run(
        "import dashscope; dashscope.Assistants",
        error_on_deprecation=True,
    )
    assert result.returncode != 0
    assert "DeprecationWarning" in result.stderr
    assert "Assistants API" in result.stderr


def test_from_import_assistants_emits_deprecation_warning():
    result = _run(
        "from dashscope import Assistants",
        error_on_deprecation=True,
    )
    assert result.returncode != 0
    assert "DeprecationWarning" in result.stderr


def test_accessing_threads_emits_deprecation_warning():
    result = _run(
        "import dashscope; dashscope.Threads",
        error_on_deprecation=True,
    )
    assert result.returncode != 0
    assert "DeprecationWarning" in result.stderr


def test_direct_assistants_import_emits_deprecation_warning():
    result = _run(
        "import dashscope.legacy_1x.assistants",
        error_on_deprecation=True,
    )
    assert result.returncode != 0
    assert "DeprecationWarning" in result.stderr


def test_lazy_names_resolve_to_real_objects():
    import dashscope
    from dashscope.legacy_1x.assistants import Assistant, Assistants
    from dashscope.legacy_1x.assistants.assistant_types import (
        AssistantFile,
        DeleteResponse,
    )
    from dashscope.legacy_1x.threads import RunStepList, Threads

    assert dashscope.Assistants is Assistants
    assert dashscope.Assistant is Assistant
    assert dashscope.AssistantFile is AssistantFile
    assert dashscope.DeleteResponse is DeleteResponse
    assert dashscope.Threads is Threads
    assert dashscope.RunStepList is RunStepList


def test_dir_includes_lazy_names():
    import dashscope

    listing = dir(dashscope)
    for name in ("Assistants", "Threads", "Runs", "Messages"):
        assert name in listing


def test_unknown_attribute_raises_attribute_error():
    import dashscope

    try:
        dashscope.ThisNameDoesNotExist
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")
