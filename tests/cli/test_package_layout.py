# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Structural invariants of the independent CLI package trees.

``dashscope.cli`` serves only v2 through ``dashscope``. The 1.x tree lives in
``dashscope.legacy_1x.cli`` and has its own ``dashscope1`` entry point.
"""

import ast
import subprocess
import sys
from pathlib import Path

from tests.cli.conftest import strip_ansi_codes

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


def test_v2_argv_executes_without_loading_the_legacy_cli():
    result = _run_python(
        "import sys\n"
        "from dashscope.cli.main import main\n"
        "try:\n"
        "    main(['apps', 'completions', 'create', '--help'])\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0\n"
        "assert 'dashscope.legacy_1x.cli' not in sys.modules\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_argv_is_rejected_without_loading_the_legacy_cli():
    result = _run_python(
        "import sys\n"
        "from dashscope.cli.main import main\n"
        "try:\n"
        "    main(['generation', 'call', '--help'])\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 2\n"
        "assert 'dashscope.legacy_1x.cli' not in sys.modules\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    error = strip_ansi_codes(result.stderr)
    assert "No such command 'generation'" in error
    assert "moved to 'dashscope1'" in error


def _module_level_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


def _all_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


def test_v2_tree_never_imports_legacy_1x():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for name in _all_imports(path):
            if name.startswith("dashscope.legacy_1x"):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {name}")
    assert offenders == []


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


def test_console_script_targets_resolve():
    """Both console scripts resolve to independent CLI implementations."""
    from importlib import import_module

    v2_module = import_module("dashscope.cli.main")
    legacy_module = import_module("dashscope.legacy_1x.cli")
    assert callable(v2_module.main)
    assert callable(legacy_module.main)


def test_legacy_tree_uses_the_dashscope1_name():
    from importlib import import_module

    module = import_module("dashscope.legacy_1x.cli")
    assert module.app.info.name == "dashscope1"


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
