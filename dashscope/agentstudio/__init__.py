# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility shim; the implementation moved to ``dashscope.v2.agentstudio``.

The AgentStudio sub-SDK is owned by the 2.0 package. This module keeps the
historical import path working: every public name is re-exported, and every
submodule is aliased in ``sys.modules`` so deep imports such as
``from dashscope.agentstudio.resources.agents import Agents`` resolve to the
very same module object -- and therefore the very same classes -- as the
``dashscope.v2.agentstudio`` path.

Import from ``dashscope.v2.agentstudio`` in new code.
"""

import pkgutil
import sys

# Importing the package runs its own eager imports, so every submodule is
# already in sys.modules by the time the walk below starts.
from dashscope.v2 import agentstudio as _impl

# pylint: disable-next=wildcard-import,unused-wildcard-import
from dashscope.v2.agentstudio import *  # noqa: F401,F403

# The wildcard only carries the names in the implementation's __all__, which
# lists classes and helpers but not its submodules. Naming them here keeps
# ``from dashscope.agentstudio import exceptions`` resolvable statically; at
# runtime they are already imported by the package's own eager imports.
# pylint: disable-next=unused-import
from dashscope.v2.agentstudio import (  # noqa: F401
    client,
    constants,
    exceptions,
    pagination,
    resources,
    streaming,
    transport,
    types,
)

_V2_PACKAGE = _impl.__name__

__all__ = list(_impl.__all__)

# Alias each submodule under the old dotted path. Registering in
# sys.modules alone is not enough: the import system skips a module it
# finds there, so the attribute on the parent would never be set and
# ``dashscope.agentstudio.resources`` would raise AttributeError.
for _info in pkgutil.walk_packages(_impl.__path__, prefix=_V2_PACKAGE + "."):
    _module = sys.modules.get(_info.name)
    if _module is None:  # pragma: no cover - walk_packages imports as it goes
        continue
    _alias = __name__ + _info.name[len(_V2_PACKAGE) :]
    sys.modules.setdefault(_alias, _module)
    _parent_name, _, _leaf = _alias.rpartition(".")
    _parent = sys.modules.get(_parent_name)
    if _parent is not None:
        setattr(_parent, _leaf, _module)
