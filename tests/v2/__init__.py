# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Checks on the v2 SDK tree as a whole, rather than on one module.

These tests read the shipped source and assert structural invariants across
it -- the sync/async twins expose the same contract, that sort of thing.
They have no runtime counterpart to import, so they live apart from the flat
per-module suites in ``tests/``.
"""
