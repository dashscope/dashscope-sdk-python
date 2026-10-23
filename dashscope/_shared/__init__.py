# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Neutral leaf layer shared by the 1.x API and the 2.0 implementation.

What belongs here is anything the two stacks must agree on exactly once
per installed distribution: the process-wide session id, the mutable
client tag, the exception base class that ``except`` clauses on both
sides match against, and the validators that keep a request pointed at
``*.maas.aliyuncs.com``. Copying one of those into either stack would
silently split a single source of truth, so both stacks import from
here instead.

The only thing this package reaches outside itself is
``dashscope.version``, read defensively by ``_sdk_ident``. That module
is a bare constant with no imports of its own, and it deliberately stays
where it is: ``setup.py`` and ``docs/source/conf.py`` ``exec()`` the
file to read the version *without* importing the package, and turning
it into a re-export of something here would drag the whole SDK and its
runtime dependencies into any build environment that only wanted a
string.

What this package must never import is the 1.x business surface
(``client``, ``common``, ``finetune``, ``audio`` and the rest).
``common/utils`` reaches ``_sdk_ident`` while ``dashscope/__init__`` is
still executing, so an import back into that surface would close a cycle
through a partially initialized root. ``dashscope.version`` is safe
precisely because it has no imports of its own; the business modules
all have plenty.
"""
