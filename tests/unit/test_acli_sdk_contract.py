# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The acli <-> SDK contract, pinned so the 1.x/2.0 layout move is safe.

``dashscope/acli/`` is a one-way sync copy of the agenticCLI repo and a
neutral top-level product surface: it serves both stacks, belongs to
neither, and must not be refactored here because the next sync would
overwrite the edit. What this repo *does* own is the SDK surface acli
consumes, so that is what gets pinned.

The dependency surface is derived from acli's own AST instead of being
hardcoded. acli is re-synced from upstream, so line numbers drift and a
sync can introduce new coupling; deriving the check from the tree means
the test reports that coupling rather than silently going stale.

The load-bearing assertion is ``api_key``. ``acli/ui/tts.py`` writes the
1.x root global ``dashscope.api_key`` because the audio SDK reads it on
paths that do not consult the environment. Promoting 2.0 to the top
level rewrites ``dashscope/__init__.py``, and 2.0 takes the key as
``Client(api_key=...)`` with no root global -- dropping the attribute
would break acli's voice output at ``if not dashscope.api_key``, while
``is_available()`` (which only does ``import dashscope``) keeps
reporting True and defers the failure to the end user.

acli's deliberately separate session id is covered by
``tests/unit/test_sdk_headers.py``; it is not repeated here.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
from functools import lru_cache

import dashscope

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ACLI = _REPO / "dashscope" / "acli"
_V2 = _REPO / "dashscope" / "v2"
_ROOT_INIT = _REPO / "dashscope" / "__init__.py"

# What acli is allowed to couple to. The bare ``dashscope`` entry is here
# because acli reads and writes the root global ``dashscope.api_key``; the
# two audio modules are the deep paths the design doc calls out. Anything
# else means a sync added coupling this repo should accept consciously.
_ALLOWLIST = frozenset(
    {
        "dashscope",
        "dashscope.legacy_1x.audio.asr",
        "dashscope.legacy_1x.audio.tts_v2",
    },
)

_ACLI_EXTRAS = frozenset(
    {
        "acli",
        "acli-all",
        "acli-anthropic",
        "acli-camera",
        "acli-openai",
        "acli-voice",
    },
)


def _is_acli_internal(module: str) -> bool:
    parts = module.split(".")
    return len(parts) > 1 and parts[1] == "acli"


def _parse(root: pathlib.Path):
    """Parse every module under ``root`` (or ``root`` itself, if a file).

    A syntax error is a failure, not a skip: silently skipping would let
    an unparseable file drop out of the survey and hollow out the guard.
    """
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    parsed = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        try:
            parsed.append((path, ast.parse(source, str(path))))
        except SyntaxError as exc:
            raise AssertionError(f"{path} does not parse: {exc}") from exc
    return parsed


@lru_cache(maxsize=None)
def _parsed(root_str: str):
    """Cached ``_parse``; four helpers walk the ~38k-line acli tree."""
    return tuple(_parse(pathlib.Path(root_str)))


@lru_cache(maxsize=None)
def _imports_under(root_str: str):
    """{module: [file:line]} for absolute ``dashscope.*`` imports."""
    found = {}
    for path, tree in _parsed(root_str):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or not node.module:
                    continue
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.split(".")[0] != "dashscope":
                    continue
                found.setdefault(module, []).append(f"{path}:{node.lineno}")
    return found


@lru_cache(maxsize=None)
def _from_import_names(root_str: str):
    """{module: sorted(names)} for ``from <dashscope module> import a, b``.

    Only ``dashscope.*`` is collected: acli's own optional extras (openai,
    playwright, PIL, sounddevice) are upstream's business and are not
    installed here, so probing them would test this machine rather than
    the contract.
    """
    found = {}
    for _, tree in _parsed(root_str):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or not node.module:
                continue
            if node.module.split(".")[0] != "dashscope":
                continue
            names = [alias.name for alias in node.names if alias.name != "*"]
            if names:
                found.setdefault(node.module, set()).update(names)
    return {mod: sorted(names) for mod, names in found.items()}


def _acli_external_imports():
    imports = _imports_under(str(_ACLI))
    return {m: loc for m, loc in imports.items() if not _is_acli_internal(m)}


def _acli_external_names():
    names = _from_import_names(str(_ACLI))
    return {m: n for m, n in names.items() if not _is_acli_internal(m)}


def _root_global_writes():
    """[(attr, file:line)] where acli assigns to ``dashscope.<attr>``."""
    writes = []
    for path, tree in _parsed(str(_ACLI)):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                base = target.value
                if isinstance(base, ast.Name) and base.id == "dashscope":
                    writes.append((target.attr, f"{path}:{node.lineno}"))
    return writes


def _environ_setdefault_calls():
    """[file:line] of ``os.environ.setdefault("DASHSCOPE_API_KEY", ...)``."""
    calls = []
    for path, tree in _parsed(str(_ACLI)):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "setdefault":
                continue
            env = func.value
            if not (
                isinstance(env, ast.Attribute)
                and env.attr == "environ"
                and isinstance(env.value, ast.Name)
                and env.value.id == "os"
            ):
                continue
            key = node.args[0]
            if isinstance(key, ast.Constant) and (
                key.value == "DASHSCOPE_API_KEY"
            ):
                calls.append(f"{path}:{node.lineno}")
    return calls


# ---------------------------------------------------------------------------
# Dependency surface
# ---------------------------------------------------------------------------


def test_acli_external_dashscope_imports_match_allowlist():
    found = _acli_external_imports()
    added = sorted(set(found) - _ALLOWLIST)
    gone = sorted(_ALLOWLIST - set(found))
    assert not added and not gone, (
        "acli's coupling to the SDK changed.\n"
        f"  added by a sync (accept consciously, then extend _ALLOWLIST): "
        f"{ {m: found[m] for m in added} }\n"
        f"  no longer imported (_ALLOWLIST is stale): {gone}"
    )


def test_acli_does_not_import_the_v2_package():
    v2 = {
        m: loc
        for m, loc in _acli_external_imports().items()
        if m.split(".")[1:2] == ["v2"]
    }
    assert not v2, (
        "acli is a 1.x-side consumer today; moving it onto the 2.0 client "
        f"is upstream agenticCLI work, not this repo's: {v2}"
    )


# ---------------------------------------------------------------------------
# Audio contract
# ---------------------------------------------------------------------------


def test_every_name_acli_imports_from_the_sdk_resolves():
    names = _acli_external_names()
    assert names, "expected acli to import names from the SDK"
    broken = {}
    for module, wanted in sorted(names.items()):
        try:
            imported = importlib.import_module(module)
        except ImportError as exc:
            broken[module] = f"import failed: {exc}"
            continue
        absent = [name for name in wanted if not hasattr(imported, name)]
        if absent:
            broken[module] = absent
    assert not broken, f"SDK surface acli relies on is incomplete: {broken}"


def test_tts_v2_exposes_the_names_acli_uses():
    module = importlib.import_module("dashscope.legacy_1x.audio.tts_v2")
    for name in ("AudioFormat", "ResultCallback", "SpeechSynthesizer"):
        assert hasattr(
            module,
            name,
        ), f"dashscope.legacy_1x.audio.tts_v2.{name} is gone"


def test_asr_exposes_the_names_acli_uses():
    module = importlib.import_module("dashscope.legacy_1x.audio.asr")
    for name in ("Recognition", "RecognitionCallback"):
        assert hasattr(
            module,
            name,
        ), f"dashscope.legacy_1x.audio.asr.{name} is gone"


# ---------------------------------------------------------------------------
# The root global acli writes
# ---------------------------------------------------------------------------


def test_root_api_key_is_readable_and_writable(monkeypatch):
    assert hasattr(dashscope, "api_key"), (
        "the root package must keep exposing ``api_key``: acli's voice "
        "output reads it and then assigns to it"
    )
    monkeypatch.setattr(dashscope, "api_key", "sk-acli-contract")
    assert dashscope.api_key == "sk-acli-contract"


def test_acli_tts_still_writes_the_root_api_key():
    attrs = {attr for attr, _ in _root_global_writes()}
    assert "api_key" in attrs, (
        "acli no longer assigns ``dashscope.api_key``. If upstream moved "
        "to the environment variable alone, the root-global requirement "
        "has genuinely relaxed -- confirm that, then drop this test and "
        "the one above it."
    )


def test_acli_also_sets_the_env_var():
    assert _environ_setdefault_calls(), (
        "acli is expected to os.environ.setdefault('DASHSCOPE_API_KEY', "
        "...) alongside the root global, so both discovery paths work"
    )


# ---------------------------------------------------------------------------
# Neutrality: acli belongs to neither side
# ---------------------------------------------------------------------------


def test_v2_package_does_not_import_acli():
    hits = {
        m: loc
        for m, loc in _imports_under(str(_V2)).items()
        if _is_acli_internal(m)
    }
    assert not hits, f"2.0 must not depend on acli: {hits}"


def test_package_root_does_not_import_acli():
    hits = {
        m: loc
        for m, loc in _imports_under(str(_ROOT_INIT)).items()
        if _is_acli_internal(m)
    }
    assert not hits, f"the 1.x package root must not depend on acli: {hits}"


def test_acli_lives_at_the_top_level():
    """acli is a neutral surface: neither the 2.0 nor the legacy bucket."""
    assert _ACLI.is_dir(), "dashscope/acli/ must stay at the top level"
    moved_into_v2 = (_REPO / "dashscope" / "v2" / "acli").exists()
    moved_into_legacy = (_REPO / "dashscope" / "legacy" / "acli").exists()
    assert not moved_into_v2, "acli must not be folded into 2.0"
    assert not moved_into_legacy, "acli must not land in the legacy bucket"


def test_acli_is_not_reexported_at_the_package_root():
    assert "acli" not in dashscope.__all__
    source = _ROOT_INIT.read_text(encoding="utf-8")
    assert "acli" not in source, (
        "acli is an independently installed entry point, not part of the "
        "1.x API layer"
    )


# ---------------------------------------------------------------------------
# Entry point and packaging
# ---------------------------------------------------------------------------


def test_module_entry_point_resolvable():
    spec = importlib.util.find_spec("dashscope.acli.__main__")
    assert spec is not None, "``python -m dashscope.acli`` needs __main__.py"
    main = importlib.import_module("dashscope.acli.cli").main
    assert callable(main)


def test_setup_declares_the_acli_extras():
    tree = ast.parse(
        (_REPO / "setup.py").read_text(encoding="utf-8"),
        "setup.py",
    )
    groups = None
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "extras_require":
            groups = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
            }
    assert groups is not None, "setup.py lost its extras_require"
    missing = sorted(_ACLI_EXTRAS - groups)
    assert not missing, f"setup.py extras_require is missing: {missing}"
