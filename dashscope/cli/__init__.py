# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The dashscope CLI (v2).

Deliberately empty. ``import dashscope.cli`` runs for shell completion and
``--help``, so it must not pull in the router, Typer, or the SDK.

There is no ``main`` re-export here on purpose: :mod:`dashscope.cli.main`
is a submodule, and binding the same name on the package would make
``from dashscope.cli import main`` resolve to a function or a module
depending on import order. The console script names the module directly
(``dashscope.cli.main:main``).
"""
