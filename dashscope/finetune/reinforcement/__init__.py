# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility shim; the implementation moved to
``dashscope.v2.finetune.reinforcement``.

The agentic-RL workflow package is owned by the 2.0 SDK. This path has to
keep resolving indefinitely, because two of its consumers resolve the
module *by name* in a process this repo does not control:

- ``FC_SERVER_CLASSPATH`` is written into the ``start.sh`` that gets
  deployed to a Function Compute sandbox, where the installed dashscope
  may be any published version.
- ``examples/workspace/scripts/start_local_*.sh`` launches the reward /
  rollout server with ``python3 -m
  dashscope.finetune.reinforcement.component.server.server``, and
  ``server.py`` hands the same dotted string to ``uvicorn.run``, which
  imports it inside its worker process.

Unlike the AgentStudio shim, aliasing here cannot be eager. Nine of the
package's modules -- ``component.server.*`` and
``component.observability.genai.*`` -- are imported lazily and need
fastapi / opentelemetry, both optional. Walking the tree up front would
either miss them or promote two optional dependencies to import-time
hard ones, so a ``sys.meta_path`` finder aliases each submodule on
demand and hands back the *same module object* the v2 path uses.

Import from ``dashscope.v2.finetune.reinforcement`` in new code.
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

_OLD = __name__
_NEW = _impl.__name__

# ``app`` is the 1.x CLI's typer app and is reachable from this path
# only; the 2.0 package must not depend on the CLI. It stays lazy
# because resolving it pulls in typer and the rest of the CLI.
# pylint: disable-next=undefined-all-variable
__all__ = list(_impl.__all__) + ["app"]  # noqa: F405


def __getattr__(name):
    if name == "app":
        from dashscope.legacy_1x.cli.agentic_rl import app

        return app
    raise AttributeError(f"module {_OLD!r} has no attribute {name!r}")


def _restore_identity(module, v2_spec):
    """Put the v2 identity back on a module object both paths share.

    ``importlib._bootstrap._init_module_attrs`` stamps the spec it was
    handed onto whatever ``create_module`` returned, and that spec names
    the legacy path. The object is the v2 module, so it must keep
    reporting the v2 name: pickling, ``importlib.reload`` and anything
    comparing ``module.__spec__.name`` against the ``sys.modules`` key
    all read it.
    """
    module.__name__ = v2_spec.name
    module.__spec__ = v2_spec
    module.__loader__ = v2_spec.loader
    module.__package__ = v2_spec.parent
    module.__file__ = v2_spec.origin
    if v2_spec.submodule_search_locations is not None:
        module.__path__ = v2_spec.submodule_search_locations


class _AliasLoader:
    """Loader that hands back the already-imported v2 module.

    Not an ``importlib.abc.Loader`` subclass on purpose: the import
    system resolves loaders by duck typing, and the ABC still declares
    ``module_repr`` abstract on this interpreter even though 3.12
    removed it.
    """

    def __init__(self, v2_spec):
        self._v2_spec = v2_spec

    def create_module(self, spec):
        module = sys.modules.get(self._v2_spec.name)
        if module is None:
            module = importlib.import_module(self._v2_spec.name)
        # Binding the attribute on the parent is what makes
        # ``dashscope.finetune.reinforcement.common.utils`` work as an
        # expression and not just as an import target: the import system
        # skips a module it already finds in sys.modules and never sets
        # the attribute itself.
        parent_name, _, leaf = spec.name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)
        return module

    def exec_module(self, module):
        # Deliberately does not re-run the module body. It already ran
        # under the v2 name, and a second pass would build a duplicate
        # FastAPI app, OpenTelemetry tracer and log handler.
        _restore_identity(module, self._v2_spec)

    def get_code(self, fullname):  # pylint: disable=unused-argument
        # ``runpy`` -- and therefore ``python -m`` -- refuses a loader
        # that yields no code object, and uvicorn resolves its import
        # string through the same path.
        return self._v2_spec.loader.get_code(self._v2_spec.name)


class _Finder:
    """Resolve ``dashscope.finetune.reinforcement.*`` to the v2 modules.

    Duck-typed for the same reason as ``_AliasLoader``: ``sys.meta_path``
    entries only need a ``find_spec``.
    """

    def find_spec(  # pylint: disable=unused-argument
        self,
        fullname,
        path=None,
        target=None,
    ):
        if not fullname.startswith(_OLD + "."):
            return None
        v2_name = _NEW + fullname[len(_OLD) :]
        # find_spec resolves without importing the leaf, so a module that
        # needs fastapi or opentelemetry stays unimported until someone
        # actually asks for it.
        v2_spec = importlib.util.find_spec(v2_name)
        if v2_spec is None or v2_spec.loader is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            v2_spec.origin,
            loader=_AliasLoader(v2_spec),
            submodule_search_locations=v2_spec.submodule_search_locations,
        )


# Inserted ahead of PathFinder rather than appended: the alias inherits
# the v2 package's __path__, so PathFinder would find the real file
# there and execute it a *second* time under the legacy name, yielding
# two distinct class objects for the same source.
if not any(isinstance(entry, _Finder) for entry in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
