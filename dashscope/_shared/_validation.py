# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Host-label validation shared by the 1.x API and the 2.0 implementation.

``workspace_id`` and ``region`` are interpolated into the host of the MaaS
endpoint, so both stacks must reject the same malformed values with the
same rule. Keeping the check in one leaf module means a value that is
safe for one stack is safe for the other, and neither stack has to import
the other to get it.

Stdlib only, like the rest of ``dashscope._shared``.
"""

import re

# workspace_id and region become DNS labels of the MaaS endpoint host, so
# they must not contain characters (/, ?, #, @, :, whitespace, ...) that
# could break out of the host and redirect requests elsewhere.
# \A...\Z rather than ^...$: $ also matches before a trailing newline.
_WORKSPACE_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
# RFC 1123 hostname label: 1-63 chars, no leading/trailing hyphen. Upper
# case is allowed because DNS is case-insensitive and httpx lowercases the
# host, so rejecting it would break working callers for no security gain.
_REGION_PATTERN = re.compile(
    r"\A[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z",
)

__all__ = ["validate_region", "validate_workspace_id"]


def validate_workspace_id(workspace_id):
    """Validate that workspace_id is safe to use as a URL subdomain.

    Raises:
        ValueError: If workspace_id contains characters outside
            [A-Za-z0-9_-], is empty, or exceeds 64 characters.
    """
    if not workspace_id or not _WORKSPACE_ID_PATTERN.fullmatch(
        workspace_id,
    ):
        raise ValueError(
            f"Invalid workspace_id {workspace_id!r}: must match "
            "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        )
    return workspace_id


def validate_region(region):
    """Validate that region is safe to use as a URL subdomain.

    Deliberately a hostname-label check and not a MAAS_REGIONS whitelist:
    rejecting an unknown-but-well-formed region would break callers as soon
    as a subpackage supports a region this list has not caught up with.

    Raises:
        ValueError: If region is not a valid hostname label.
    """
    if not region or not _REGION_PATTERN.fullmatch(region):
        raise ValueError(
            f"Invalid region {region!r}: must be a hostname label "
            "(1-63 chars of [A-Za-z0-9-], not starting or ending "
            "with '-')",
        )
    return region
