# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Command groups for the dashscope CLI.

Every module here follows the same shape, which is the pattern the rest
of the command tree is meant to copy: declare options with typer, take
``ctx: typer.Context`` as the first parameter, make exactly one SDK call
through ``ctx.obj.client``, and hand the result to ``ctx.obj.renderer``.

The rule that keeps ``--help`` fast is structural: **no module in this
package may import ``dashscope`` at module scope.** Importing the SDK
pulls in pydantic, httpx and the 1.x resource tree; a user reading
``dashscope chat create --help`` should not wait for any of it. Reach the
SDK only through ``ctx.obj.client``, which is built on first use.
"""
