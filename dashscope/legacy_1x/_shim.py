# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lazy aliases for the historical 1.x import paths."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys


_APPLICATION_ROOTS = frozenset({"finetune"})

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
        "tokenizers",
        "utils",
    },
)

_EXACT_MODULES = {
    "dashscope.finetune": "dashscope.legacy_1x.finetune",
    "dashscope.agentstudio": "dashscope.legacy_1x.agentstudio",
    "dashscope.agentstudio.client": "dashscope.legacy_1x.agentstudio.client",
    "dashscope.agentstudio.exceptions": (
        "dashscope.legacy_1x.agentstudio.exceptions"
    ),
    "dashscope.agentstudio.streaming": (
        "dashscope.legacy_1x.agentstudio.streaming"
    ),
    "dashscope.agentstudio.transport": (
        "dashscope.legacy_1x.agentstudio.transport"
    ),
    "dashscope.files": "dashscope.legacy_1x.files",
    "dashscope.model": "dashscope.legacy_1x.model",
    "dashscope.models": "dashscope.legacy_1x.models",
    "dashscope.finetune.agentic_rl": (
        "dashscope.legacy_1x.finetune.agentic_rl"
    ),
    "dashscope.finetune.reinforcement": (
        "dashscope.legacy_1x.finetune.reinforcement"
    ),
    "dashscope.finetune.reinforcement.common": (
        "dashscope.legacy_1x.finetune.reinforcement.common"
    ),
    "dashscope.finetune.reinforcement.common.errors": (
        "dashscope.legacy_1x.finetune.reinforcement.common.errors"
    ),
    "dashscope.finetune.customize_types": (
        "dashscope.legacy_1x.finetune.customize_types"
    ),
    "dashscope.finetune.deployments": (
        "dashscope.legacy_1x.finetune.deployments"
    ),
    "dashscope.finetune.finetunes": "dashscope.legacy_1x.finetune.finetunes",
    # ``dashscope.finetune`` aliases the shared ``dashscope.applications.finetune``
    # module object, so ``from dashscope.finetune import <sub>`` resolves the child
    # against that canonical ``__name__`` rather than the alias prefix above. Map the
    # canonical names too, otherwise those 1.x-only submodules raise ImportError.
    "dashscope.applications.finetune.customize_types": (
        "dashscope.legacy_1x.finetune.customize_types"
    ),
    "dashscope.applications.finetune.deployments": (
        "dashscope.legacy_1x.finetune.deployments"
    ),
    "dashscope.applications.finetune.finetunes": (
        "dashscope.legacy_1x.finetune.finetunes"
    ),
}

_FINDER_MARKER = "_dashscope_legacy_1x_alias_finder"

_CLIENT_NAME = "dashscope.client"


def _client_module_spec():
    """Bind ``dashscope.client`` to ``client.py`` when a leftover 1.x ``client/``
    directory shadows it, since CPython resolves directories first.

    Returns None on a clean install, where normal resolution already works.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_file = os.path.join(root, "client.py")
    if not os.path.isfile(module_file):
        return None
    if not os.path.isdir(os.path.join(root, "client")):
        return None
    return importlib.util.spec_from_file_location(_CLIENT_NAME, module_file)


def _canonical_name(name):
    if name in _EXACT_MODULES:
        return _EXACT_MODULES[name]
    if name.startswith("dashscope.client."):
        return "dashscope.legacy_1x.client." + name[len("dashscope.client.") :]
    if name.startswith("dashscope.agentstudio."):
        suffix = name[len("dashscope.agentstudio.") :]
        return "dashscope.applications.agentstudio." + suffix
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
        if fullname == _CLIENT_NAME:
            return _client_module_spec()
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
