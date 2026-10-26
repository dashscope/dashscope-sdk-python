# -*- coding: utf-8 -*-
"""
component/parser/base_parser.py

Base class definitions for request parameter parsing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from dashscope.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
)


class BaseRequestParser(ABC):
    """
    Base class for request parameter parsing.
    Each business type must implement parse() method to convert raw request
    dict
    into corresponding BaseDataModel subclass instance.
    """

    @abstractmethod
    def parse(self, raw: Dict[str, Any]) -> BaseDataModel:
        """
        Convert raw request body (dict) into processor input object.

        Args:
            raw: Raw dictionary from HTTP request body

        Returns:
            Concrete BaseDataModel subclass instance
        """
        raise NotImplementedError
