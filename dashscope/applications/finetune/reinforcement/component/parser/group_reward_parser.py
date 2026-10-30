# -*- coding: utf-8 -*-
"""
component/parser/group_reward_parser.py

Request parameter parser for GroupReward business type.
"""

from typing import Any, Dict

from ...common.log import logger
from ..data import (
    GroupRewardInput,
)
from .base_parser import (
    BaseRequestParser,
)


class GroupRewardRequestParser(BaseRequestParser):
    """
    Request parameter parser for GroupReward business type.
    Converts raw HTTP request dict into GroupRewardInput instance.

    Note: This is framework-internal class, not customizable by users.
    """

    def parse(self, raw: Dict[str, Any]) -> GroupRewardInput:
        """
        Parse GroupReward request parameters.

        Args:
            raw: Raw HTTP request body dict

        Returns:
            GroupRewardInput instance

        Raises:
            ValueError: When required fields are missing or invalid
        """
        logger.debug(f"[GroupRewardRequestParser] parsing raw request: {raw}")
        return GroupRewardInput(**raw)
