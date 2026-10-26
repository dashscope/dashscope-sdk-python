# -*- coding: utf-8 -*-
"""
component/parser/reward_parser.py

Request parameter parser for Reward business type.
"""

from typing import Any, Dict

from dashscope.applications.finetune.reinforcement.common.log import logger
from dashscope.applications.finetune.reinforcement.component.data.reward_input import (
    RewardInput,
)
from dashscope.applications.finetune.reinforcement.component.parser.base_parser import (
    BaseRequestParser,
)


class RewardRequestParser(BaseRequestParser):
    """
    Request parameter parser for Reward business type.
    Converts raw HTTP request dict into RewardInput instance.

    Note: This is framework-internal class, not customizable by users.
    """

    def parse(self, raw: Dict[str, Any]) -> RewardInput:
        """
        Parse Reward request parameters.

        Args:
            raw: Raw HTTP request body dict

        Returns:
            RewardInput instance

        Raises:
            ValueError: When required fields are missing or invalid
        """
        logger.debug(f"[RewardRequestParser] parsing raw request: {raw}")
        return RewardInput(**raw)
