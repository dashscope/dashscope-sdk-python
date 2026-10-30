# -*- coding: utf-8 -*-
"""
component/parser/rollout_parser.py

Request parameter parser for Rollout business type.
"""

from typing import Any, Dict

from ...common.log import logger
from ..data.rollout_input import (
    RolloutInput,
)
from .base_parser import (
    BaseRequestParser,
)


class RolloutRequestParser(BaseRequestParser):
    """
    Request parameter parser for Rollout business type.
    Converts raw HTTP request dict into RolloutInput instance.

    Note: This is framework-internal class, not customizable by users.
    """

    def parse(self, raw: Dict[str, Any]) -> RolloutInput:
        """
        Parse Rollout request parameters.

        Args:
            raw: Raw HTTP request body dict

        Returns:
            RolloutInput instance

        Raises:
            ValueError: When required fields are missing or invalid
        """
        logger.debug(f"[RolloutRequestParser] parsing raw request: {raw}")
        return RolloutInput(**raw)
