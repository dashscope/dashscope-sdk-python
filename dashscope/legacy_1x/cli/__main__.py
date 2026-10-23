# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
"""Allow ``python -m dashscope.legacy_1x.cli`` invocation."""
import warnings

# Suppress urllib3 NotOpenSSLWarning on systems with LibreSSL
warnings.filterwarnings(
    "ignore",
    message=".*urllib3.*only supports OpenSSL.*",
    category=Warning,
)

# fmt: off
from dashscope.legacy_1x.cli import main  # noqa: E402
# fmt: on

main()
