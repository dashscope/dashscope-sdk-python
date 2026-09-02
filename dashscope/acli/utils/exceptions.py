# -*- coding: utf-8 -*-
"""Common exception classes used across the CLI."""

from __future__ import annotations


class UserAbortedTurn(Exception):
    """Raised when the user picks [s]top at a confirmation prompt — signals
    the agent loop to bail out entirely (not just skip one tool call)."""


class UserSupplement(Exception):
    """Raised when the user picks [u]pdate at a confirmation prompt —
    signals the agent loop to inject supplemental info into the
    conversation and re-plan before continuing tool execution."""

    def __init__(self, supplement: str):
        self.supplement = supplement
        super().__init__(supplement)
