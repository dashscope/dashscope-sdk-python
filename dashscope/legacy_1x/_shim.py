# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Backward-compat aliasing for the historical top-level 1.x import paths.

The 1.x business surface now lives under ``dashscope.legacy_1x``. The thin
shim packages left at the old top-level names (``dashscope/aigc/__init__.py``
and friends) call :func:`install` so that ``import dashscope.aigc`` and
``from dashscope.aigc.generation import Generation`` keep resolving.

The shim re-exports the implementation's public names and points its
``__path__`` at the implementation package, so a submodule import loads the
real file. Submodules reached through the old spelling are therefore distinct
module objects from their ``dashscope.legacy_1x.*`` twins; nothing inside this
repository relies on that (all internal code, tests and examples use the
canonical ``dashscope.legacy_1x`` path, and ``import dashscope`` re-exports the
public names directly), so the aliasing is best-effort back-compat for external
callers only. It is deliberately lazy -- it never imports submodules itself --
so importing a shim cannot trigger import-time side effects in the 1.x tree
(the ``dashscope.legacy_1x.cli`` package has some).
"""

from __future__ import annotations

import importlib
import sys


def install(shim_name: str, impl_name: str):
    """Alias the package/module ``shim_name`` onto ``impl_name``.

    Returns the implementation module so a shim can also do
    ``_impl = install(__name__, "dashscope.legacy_1x.aigc")`` when it wants
    to reference it.
    """
    impl = importlib.import_module(impl_name)
    shim = sys.modules[shim_name]

    # Some 1.x packages declare a malformed ``__all__`` (e.g. ``audio``
    # lists module objects rather than names), so union the string entries
    # of ``__all__`` with every public attribute instead of trusting it.
    declared = getattr(impl, "__all__", None) or []
    names = {name for name in declared if isinstance(name, str)}
    names.update(name for name in dir(impl) if not name.startswith("_"))
    for name in names:
        try:
            setattr(shim, name, getattr(impl, name))
        except AttributeError:  # pragma: no cover - name vanished mid-walk
            continue

    if hasattr(impl, "__path__"):
        # Let ``import dashscope.<shim>.<sub>`` find the implementation's
        # files without eagerly importing them here.
        shim.__path__ = impl.__path__

    return impl
