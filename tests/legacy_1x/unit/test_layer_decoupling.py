# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Import-graph boundaries for the root implementation and legacy APIs.

``dashscope/_shared`` is the neutral layer both stacks import from. The
current implementation must not reach into the 1.x business surface, apart
from explicitly pinned compatibility adapters. These are import-graph
properties, so they are surveyed as AST rather than left as prose.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[3]
_DASHSCOPE = _REPO / "dashscope"
_SHARED = _DASHSCOPE / "_shared"
_CURRENT = (
    _DASHSCOPE / "_config.py",
    _DASHSCOPE / "_error_registry.py",
    _DASHSCOPE / "_logging.py",
    _DASHSCOPE / "_quickstart.py",
    _DASHSCOPE / "finetune" / "agentic_rl.py",
    _DASHSCOPE / "client.py",
    _DASHSCOPE / "errors.py",
    _DASHSCOPE / "finetune" / "_submit.py",
    _DASHSCOPE / "protocols",
    _DASHSCOPE / "resources",
    _DASHSCOPE / "transport",
    _DASHSCOPE / "types",
    _DASHSCOPE / "agentstudio",
    _DASHSCOPE / "finetune" / "reinforcement",
    _DASHSCOPE / "tokenizers" / "tokenizer.py",
    _DASHSCOPE / "tokenizers" / "tokenizer_base.py",
    _DASHSCOPE / "tokenizers" / "qwen_tokenizer.py",
)

_VERSION_FILE = "dashscope/version.py"

_NEUTRAL_PACKAGES = frozenset(
    {
        "_shared",
        "acli",
        "agentstudio",
        "finetune",
        "protocols",
        "resources",
        "transport",
        "types",
        "tokenizers",
    },
)

_NEUTRAL_LEAVES = frozenset(
    {
        "dashscope",
        "dashscope.version",
        "dashscope._config",
        "dashscope._error_registry",
        "dashscope._logging",
        "dashscope._quickstart",
        "dashscope.client",
        "dashscope.errors",
        "dashscope.finetune._submit",
    },
)

_SKIP_PARTS = frozenset({"examples"})

_COMPATIBILITY_IMPORTS = {
    (
        "dashscope/finetune/reinforcement/__init__.py",
        "dashscope.legacy_1x.cli.agentic_rl",
    ),
}

# ``__version__`` bindings that are not the distribution version.
_OTHER_VERSION_OWNERS = frozenset(
    {
        # acli versions itself and is a one-way sync copy of the
        # agenticCLI repo, so it must stay independent.
        "dashscope/acli/__init__.py",
        # The version of the vendored error-registry data.
        "dashscope/_error_registry.py",
    },
)

_DISABLE_ENV = "DASHSCOPE_DISABLE_SDK_HEADERS"


def _parse(path: pathlib.Path) -> ast.AST:
    """Parse one module. A syntax error fails rather than skips."""
    source = path.read_text(encoding="utf-8")
    try:
        return ast.parse(source, str(path))
    except SyntaxError as exc:
        raise AssertionError(f"{path} does not parse: {exc}") from exc


def _modules(root: pathlib.Path):
    assert root.exists(), f"scan target does not exist: {root}"
    if root.is_file():
        return [root]
    paths = [
        path
        for path in sorted(root.rglob("*.py"))
        if not _SKIP_PARTS.intersection(path.parts)
    ]
    assert paths, f"scan target contains no Python modules: {root}"
    return paths


def _dashscope_imports(tree: ast.AST):
    """[(module, lineno)] for absolute ``dashscope.*`` imports."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.append((node.module, node.lineno))
    return [
        (module, lineno)
        for module, lineno in found
        if module.split(".")[0] == "dashscope"
    ]


def _is_neutral(module: str) -> bool:
    if module in _NEUTRAL_LEAVES:
        return True
    parts = module.split(".")
    return len(parts) > 1 and parts[1] in _NEUTRAL_PACKAGES


def _module_level_version_owners(tree: ast.AST):
    """Line numbers of module-scope ``__version__ =`` assignments.

    Function-scope bindings are deliberately excluded. Two readers --
    ``transport/http.default_user_agent`` and
    ``agentstudio/client._user_agent`` -- degrade with a local
    ``except Exception: __version__ = "0.0.0"``. That rebinds a name
    inside one function; it does not define the distribution version.
    """
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(t, ast.Name) and t.id == "__version__"
            for t in node.targets
        ):
            found.append(node.lineno)
    return found


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(_REPO).as_posix()


def test_current_implementation_does_not_import_1x_business_surface():
    offenders = []
    scanned = []
    for root in _CURRENT:
        for path in _modules(root):
            scanned.append(path)
            for module, lineno in _dashscope_imports(_parse(path)):
                if _is_neutral(module):
                    continue
                if (_rel(path), module) in _COMPATIBILITY_IMPORTS:
                    continue
                offenders.append(f"{_rel(path)}:{lineno} -> {module}")
    assert scanned, "the current implementation scan must not be empty"
    assert not offenders, (
        "the current implementation must not import the 1.x business "
        "surface outside explicit compatibility adapters:\n  "
        + "\n  ".join(offenders)
    )


def test_shared_layer_imports_no_1x_business_surface():
    """``_shared`` stays below both stacks, or it cannot anchor them.

    ``common/utils.py`` reaches ``_shared._sdk_ident`` while
    ``dashscope/__init__`` is still executing, so a ``_shared`` module
    that imports the 1.x surface closes a cycle through a partially
    initialized root. ``dashscope.version`` is the one allowed exception
    and is safe only because it imports nothing -- see
    ``test_version_file_has_no_imports``.
    """
    offenders = []
    for path in _modules(_SHARED):
        for module, lineno in _dashscope_imports(_parse(path)):
            if _is_neutral(module):
                continue
            offenders.append(f"{_rel(path)}:{lineno} -> {module}")
    assert not offenders, (
        "dashscope/_shared may import only the stdlib, itself, and "
        "dashscope.version:\n  " + "\n  ".join(offenders)
    )


def test_version_file_has_no_imports():
    """``dashscope/version.py`` must stay readable without importing.

    ``setup.py`` and ``docs/source/conf.py`` ``exec()`` this file to read
    the version rather than importing it, which is what lets packaging
    work in an isolated build environment where the runtime dependencies
    are not installed. Any import added here -- a re-export of the value
    from elsewhere, for instance -- turns reading one string into loading
    the whole SDK and its dependencies.
    """
    imports = [
        node.lineno
        for node in ast.walk(_parse(_DASHSCOPE / "version.py"))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not imports, (
        f"{_VERSION_FILE} is exec'd standalone by setup.py and "
        f"docs/source/conf.py, so it must not import anything; found "
        f"imports at lines {imports}"
    )


def test_distribution_version_is_assigned_in_one_place():
    """One distribution version, defined once.

    A second literal would let the wheel's version and the version
    reported in request headers drift apart, since ``setup.py`` reads the
    file directly while the SDK reads the imported value. Only
    module-scope bindings count; ``_OTHER_VERSION_OWNERS`` lists the two
    that mean something else.
    """
    owners = []
    for path in _modules(_DASHSCOPE):
        rel = _rel(path)
        if rel in _OTHER_VERSION_OWNERS:
            continue
        for lineno in _module_level_version_owners(_parse(path)):
            owners.append((rel, lineno))
    located = [f"{rel}:{lineno}" for rel, lineno in owners]
    assert [rel for rel, _ in owners] == [_VERSION_FILE], (
        "the distribution version must be assigned only in "
        f"{_VERSION_FILE}; found: {located}"
    )


def test_one_version_reaches_every_identity_string(monkeypatch):
    """Every request identity carries the same version segment.

    Each of these four sites reads the version through its own
    defensive fallback to ``"0.0.0"``, so a read that failed would
    degrade one telemetry string instead of failing loudly -- and the
    header and the user-agent would disagree within one process. This
    pins that they agree, including the two user-agents whose version
    segment no other test looks at.
    """
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    version = importlib.import_module("dashscope.version").__version__

    root = importlib.import_module("dashscope")
    assert root.__version__ == version

    ident = importlib.import_module("dashscope._shared._sdk_ident")
    headers = ident.get_sdk_headers()
    assert headers["x-dashscope-sdk-client"].split("/")[1] == version
    assert ident.get_user_agent().startswith(f"dashscope/{version};")

    v2_ident = importlib.import_module("dashscope.transport._ident")
    assert v2_ident.SDK_CLIENT_VALUE.split("/")[1] == version

    http = importlib.import_module("dashscope.transport.http")
    assert http.default_user_agent() == f"dashscope-v2-python/{version}"

    studio = importlib.import_module("dashscope.agentstudio.client")
    agent = studio._user_agent("https://agentstudio.invalid")
    assert agent == (
        f"dashscope-agentstudio-python/{version} "
        "(+https://agentstudio.invalid)"
    )
