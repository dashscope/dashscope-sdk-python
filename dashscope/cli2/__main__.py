# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``python -m dashscope.cli2``.

The 1.x tree owns ``python -m dashscope.cli`` and must keep it, so the v2
entry point is spelled with the same ``2`` the package uses.
"""

from dashscope.cli2.main import main

if __name__ == "__main__":
    main()
