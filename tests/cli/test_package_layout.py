# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Structural invariants of the CLI package layout.

``dashscope.cli`` is the v2 CLI. The 1.x command tree lives at
``dashscope.legacy_1x.cli`` and is reachable through its own console
script (``dashscope1``); ``dashscope`` itself serves the v2 tree only and
rejects 1.x command names, pointing at whatever replaced them.
"""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "dashscope" / "cli"
COMMANDS_ROOT = PACKAGE_ROOT / "commands"


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_never_loads_the_1x_command_tree():
    """What importing ``dashscope.cli`` can and cannot cost.

    ``dashscope/__init__.py`` eagerly imports ``dashscope.client`` *and*
    the whole 1.x resource stack, so pydantic, httpx and ``legacy_1x.aigc``
    are paid by any ``dashscope.*`` import; no subpackage layout avoids
    that. What this package does avoid is ``dashscope.legacy_1x.cli`` --
    18 Typer submodules plus ``set_sdk_client`` -- which is the part that
    would otherwise make every ``dashscope --help`` slow and would also
    mutate the 1.x global SDK identity for v2-only invocations.
    """
    result = _run_python(
        "import sys, dashscope.cli, dashscope.cli.context\n"
        "import dashscope.cli.commands\n"
        "assert 'dashscope.legacy_1x.cli' not in sys.modules\n"
        "assert 'dashscope.client' in sys.modules\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _module_level_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


def test_command_modules_never_import_the_sdk_at_module_scope():
    """The iron rule: SDK access goes through ``ctx.obj.client``.

    Enforced by reading the AST rather than by convention, because a
    single stray top-level ``from dashscope import Client`` in any future
    command module would make every ``--help`` pay for pydantic and httpx.

    ``dashscope.cli`` itself is allowed: importing the package runs an
    ``__init__.py`` that is required to be empty (see
    :func:`test_package_init_stays_lazy`).
    """
    offenders = []
    for path in sorted(COMMANDS_ROOT.glob("*.py")):
        for name in _module_level_imports(path):
            if name == "dashscope" or (
                name.startswith("dashscope.")
                and name != "dashscope.cli"
                and not name.startswith("dashscope.cli.")
            ):
                offenders.append(f"{path.name}: {name}")
    assert offenders == []


def test_console_script_target_resolves():
    """``setup.py`` points at ``dashscope.cli.main:main``.

    ``main`` there is a *module*, so the package must not bind the same
    name to a function: whichever import ran first would win, and
    ``from dashscope.cli import main`` would mean two different things.
    """
    from importlib import import_module

    module = import_module("dashscope.cli.main")
    assert callable(module.main)


def test_package_init_stays_lazy():
    """``__init__.py`` imports nothing at all.

    It runs for shell completion and ``--help``, so anything reaching
    back into ``dashscope`` would drag the SDK (or the 1.x tree) in.
    """
    offenders = [
        name
        for name in _module_level_imports(PACKAGE_ROOT / "__init__.py")
        if name == "dashscope" or name.startswith("dashscope.")
    ]
    assert offenders == []
