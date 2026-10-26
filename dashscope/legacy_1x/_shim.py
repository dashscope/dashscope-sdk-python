# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lazy aliases for the historical 1.x import paths."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


_APPLICATION_ROOTS = frozenset({"agentstudio", "finetune"})

_PACKAGE_ROOTS = frozenset(
    {
        "aigc",
        "api_entities",
        "app",
        "assistants",
        "audio",
        "cli",
        "common",
        "embeddings",
        "io",
        "multimodal",
        "nlp",
        "protocol",
        "rerank",
        "threads",
        "utils",
    },
)

_EXACT_MODULES = {
    "dashscope.files": "dashscope.legacy_1x.files",
    "dashscope.model": "dashscope.legacy_1x.model",
    "dashscope.models": "dashscope.legacy_1x.models",
    "dashscope.finetune.customize_types": (
        "dashscope.legacy_1x.finetune.customize_types"
    ),
    "dashscope.finetune.deployments": (
        "dashscope.legacy_1x.finetune.deployments"
    ),
    "dashscope.finetune.finetunes": "dashscope.legacy_1x.finetune.finetunes",
}

_FINDER_MARKER = "_dashscope_legacy_1x_alias_finder"


def _canonical_name(name):
    if name in _EXACT_MODULES:
        return _EXACT_MODULES[name]
    if name.startswith("dashscope.client."):
        return "dashscope.legacy_1x.client." + name[len("dashscope.client.") :]
    parts = name.split(".")
    if len(parts) > 1 and parts[0] == "dashscope":
        if parts[1] in _APPLICATION_ROOTS:
            return "dashscope.applications." + ".".join(parts[1:])
        if parts[1] in _PACKAGE_ROOTS:
            return "dashscope.legacy_1x." + ".".join(parts[1:])
    return None


# pylint: disable=abstract-method,unused-argument
class _AliasLoader(importlib.abc.Loader):
    def __init__(self, alias_name, canonical_name, canonical_spec):
        self._alias_name = alias_name
        self._canonical_name = canonical_name
        self._canonical_spec = canonical_spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[self._alias_name] = importlib.import_module(
            self._canonical_name,
        )

    def get_code(self, fullname):
        loader = self._canonical_spec.loader
        if loader is None or not hasattr(loader, "get_code"):
            return None
        return loader.get_code(self._canonical_name)


# pylint: disable=unused-argument
class _AliasFinder(importlib.abc.MetaPathFinder):
    _dashscope_legacy_1x_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        canonical_name = _canonical_name(fullname)
        if canonical_name is None:
            return None
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None
        loader = _AliasLoader(fullname, canonical_name, canonical_spec)
        is_package = canonical_spec.submodule_search_locations is not None
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=canonical_spec.origin,
            is_package=is_package,
        )
        if is_package:
            spec.submodule_search_locations = (
                canonical_spec.submodule_search_locations
            )
        return spec


def install():
    """Install the process-wide lazy aliases without importing business modules."""
    for finder in sys.meta_path:
        if getattr(finder, _FINDER_MARKER, False):
            return
    sys.meta_path.insert(0, _AliasFinder())


__all__ = ["install"]
