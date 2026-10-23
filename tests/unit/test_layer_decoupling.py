# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The layer boundaries the 1.x / 2.0 decoupling rests on.

``dashscope/_shared`` is the neutral layer both stacks import from, and
the 2.0 package must not reach into the 1.x *business* surface at all.
Neither rule is enforced by the interpreter: both are import-graph
properties that a well-meaning edit undoes silently, and both only fail
at the moment the 1.x tree moves to ``legacy_1x/`` and 2.0 is promoted to
the top level. So they are surveyed here as AST rather than left as
prose.

``test_v2_does_not_import_the_1x_business_surface`` is where the design
doc's "hits must decrease monotonically" principle lands: zero. That
zero is the precondition for the physical move, not a consequence of it.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DASHSCOPE = _REPO / "dashscope"
_SHARED = _DASHSCOPE / "_shared"
_V2 = _DASHSCOPE / "v2"

_VERSION_FILE = "dashscope/version.py"

# Second segment of a ``dashscope.*`` module that is not 1.x surface.
# ``acli`` serves both stacks and stays at the top level; ``_shared`` is
# the neutral layer itself.
_NEUTRAL_PACKAGES = frozenset({"v2", "_shared", "acli"})

# ``dashscope.version`` is distribution metadata, not 1.x API surface: it
# is a bare constant module with no imports of its own, so it does not
# move to ``legacy_1x/`` and importing it cannot close a cycle. It is read by
# the neutral layer and by the three v2 identity builders.
_NEUTRAL_LEAVES = frozenset({"dashscope.version"})

# Reward-server payloads that live under v2 but are deliberately not
# packaged. They import the old deep paths on purpose: once the layout
# move makes those paths canonical again the shims disappear and these
# files are correct without being edited.
_SKIP_PARTS = frozenset({"examples"})

# ``__version__`` bindings that are not the distribution version.
_OTHER_VERSION_OWNERS = frozenset(
    {
        # acli versions itself and is a one-way sync copy of the
        # agenticCLI repo, so it must stay independent.
        "dashscope/acli/__init__.py",
        # The version of the vendored error-registry data.
        "dashscope/v2/_error_registry.py",
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
    """Every module under ``root``, minus the excluded subtrees."""
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not _SKIP_PARTS.intersection(path.parts)
    ]


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

    Function-scope bindings are deliberately excluded. Two of the four
    version readers -- ``v2/transport/http.default_user_agent`` and
    ``v2/agentstudio/client._user_agent`` -- degrade with a local
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


def test_v2_does_not_import_the_1x_business_surface():
    """2.0 depends on the neutral layer and on nothing else in 1.x.

    This is the decreasing gate the decoupling is measured by, and the
    precondition for moving 1.x to ``legacy_1x/``: every hit here is a deep
    path that stops resolving the moment the tree it points at moves.
    """
    offenders = []
    for path in _modules(_V2):
        for module, lineno in _dashscope_imports(_parse(path)):
            if _is_neutral(module):
                continue
            offenders.append(f"{_rel(path)}:{lineno} -> {module}")
    assert not offenders, (
        "2.0 must not import the 1.x business surface; these would break "
        "the moment 1.x moves to legacy_1x/:\n  " + "\n  ".join(offenders)
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

    v2_ident = importlib.import_module("dashscope.v2.transport._ident")
    assert v2_ident.SDK_CLIENT_VALUE.split("/")[1] == version

    http = importlib.import_module("dashscope.v2.transport.http")
    assert http.default_user_agent() == f"dashscope-v2-python/{version}"

    studio = importlib.import_module("dashscope.v2.agentstudio.client")
    agent = studio._user_agent("https://agentstudio.invalid")
    assert agent == (
        f"dashscope-agentstudio-python/{version} "
        "(+https://agentstudio.invalid)"
    )
