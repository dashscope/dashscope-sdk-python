# -*- coding: utf-8 -*-
"""
component/processor/abstract_processor.py

Abstract base class definitions for business processing.
"""

from abc import ABC, abstractmethod
from typing import Any

from dashscope.finetune.reinforcement.component.data.base_data_model import (
    BaseDataModel,
)


class AbstractProcessor(ABC):
    """
    Abstract base class for business processing.
    Each business type should inherit this class and
    implement the process() method.
    """

    def setup(self) -> None:
        """
        Initialize workspace before the server starts processing requests.

        Override this method to perform one-time initialization tasks such as:
        - Loading ML models into memory
        - Establishing database connections
        - Initializing cache
        - Loading configuration files
        - Pre-computing static data

        This method is called once during server startup, before any
        requests are processed.
        The default implementation does nothing.

        Raises:
            Exception: If initialization fails, the server will fail to start.
        """

    @abstractmethod
    async def process(self, input_data: BaseDataModel) -> Any:
        """
        Execute business logic and return results.

        Args:
            input_data: Parsed business input object (subclass of
            BaseDataModel).

        Returns:
            Any serializable result object that will be returned to the caller.
        """
        raise NotImplementedError
