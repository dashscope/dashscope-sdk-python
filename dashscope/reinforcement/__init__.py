# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility shim; the implementation lives in
``dashscope.v2.finetune.reinforcement``.

This is the public reinforcement import path. The former
``dashscope.finetune.reinforcement`` path registers itself as an additional
alias so existing Function Compute sandboxes and local launch scripts keep
working.

Aliasing cannot be eager. Nine of the package's modules --
``component.server.*`` and ``component.observability.genai.*`` -- are
imported lazily and need fastapi / opentelemetry, both optional. Walking
the tree up front would either miss them or promote two optional
dependencies to import-time hard ones, so a ``sys.meta_path`` finder aliases
each submodule on demand and hands back the *same module object* the v2 path
uses.
"""

import importlib
import importlib.util
import sys

# Importing the package runs its own dependency guard, so a missing
# pydantic / tenacity / pyyaml still raises from this path.
from dashscope.v2.finetune import reinforcement as _impl

# pylint: disable-next=wildcard-import,unused-wildcard-import
from dashscope.v2.finetune.reinforcement import *  # noqa: F401,F403

# The wildcard only carries the names in the implementation's __all__,
# which lists classes and helpers but not its subpackages. Binding them
# here keeps the deep imports the samples use statically resolvable for
# pylint and IDEs, to whom the meta_path finder below is invisible. At
# runtime both are already imported by the package's own eager imports.
# pylint: disable-next=unused-import
from dashscope.v2.finetune.reinforcement import (  # noqa: F401
    common,
    component,
)

_PUBLIC = __name__
_IMPL = _impl.__name__

# ``app`` is the 1.x CLI's typer app and is reachable from the compatibility
# paths only; the 2.0 package must not depend on the CLI. It stays lazy
# because resolving it pulls in typer and the rest of the CLI.
# pylint: disable-next=undefined-all-variable
__all__ = list(_impl.__all__) + ["app"]  # noqa: F405


def __getattr__(name):
    if name == "app":
        from dashscope.legacy_1x.cli.agentic_rl import app

        return app
    raise AttributeError(f"module {_PUBLIC!r} has no attribute {name!r}")


def _restore_identity(module, v2_spec):
    """Put the v2 identity back on a module object all paths share."""
    module.__name__ = v2_spec.name
    module.__spec__ = v2_spec
    module.__loader__ = v2_spec.loader
    module.__package__ = v2_spec.parent
    module.__file__ = v2_spec.origin
    if v2_spec.submodule_search_locations is not None:
        module.__path__ = v2_spec.submodule_search_locations


class _AliasLoader:
    """Hand back the already-imported v2 module without executing it again."""

    def __init__(self, v2_spec):
        self._v2_spec = v2_spec

    def create_module(self, spec):
        module = sys.modules.get(self._v2_spec.name)
        if module is None:
            module = importlib.import_module(self._v2_spec.name)
        # The import system skips a module already in sys.modules, so bind
        # the leaf on the parent for attribute-expression imports.
        parent_name, _, leaf = spec.name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)
        return module

    def exec_module(self, module):
        _restore_identity(module, self._v2_spec)

    def get_code(self, fullname):  # pylint: disable=unused-argument
        # runpy (python -m) and uvicorn require a loader with a code object.
        return self._v2_spec.loader.get_code(self._v2_spec.name)


class _Finder:
    """Resolve one public package prefix to the v2 modules."""

    def __init__(self, alias_name):
        self.alias_name = alias_name

    def find_spec(  # pylint: disable=unused-argument
        self,
        fullname,
        path=None,
        target=None,
    ):
        if not fullname.startswith(self.alias_name + "."):
            return None
        v2_name = _IMPL + fullname[len(self.alias_name) :]
        # find_spec resolves without importing the leaf, preserving the
        # optional fastapi and opentelemetry dependencies.
        v2_spec = importlib.util.find_spec(v2_name)
        if v2_spec is None or v2_spec.loader is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            v2_spec.origin,
            loader=_AliasLoader(v2_spec),
            submodule_search_locations=v2_spec.submodule_search_locations,
        )


def _register_alias(alias_name):
    # Register ahead of PathFinder so it cannot execute the v2 source again
    # under the compatibility name and create duplicate class objects.
    if not any(
        isinstance(entry, _Finder) and entry.alias_name == alias_name
        for entry in sys.meta_path
    ):
        sys.meta_path.insert(0, _Finder(alias_name))


_register_alias(_PUBLIC)
