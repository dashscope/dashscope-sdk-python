# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agentic fine-tuning, owned by the 2.0 package.

Holds the agentic-RL workflow package (``reinforcement``) and the
``AgenticRL`` driver that ``Client(...).rl`` is built on.

The 1.x job-management surface -- ``dashscope.FineTunes`` and
``dashscope.Deployments`` -- stays at ``dashscope.finetune``; only the
modules listed in the decoupling plan moved here, and each keeps a
reverse-reference shim at its old import path.

Deliberately imports nothing at module level: ``reinforcement`` requires
pydantic / tenacity / pyyaml, and pulling those into ``import
dashscope.v2`` would make the 2.0 client depend on them.
"""
