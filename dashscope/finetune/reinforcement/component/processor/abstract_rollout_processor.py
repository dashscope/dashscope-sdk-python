# -*- coding: utf-8 -*-
"""
component/processor/abstract_rollout_processor.py

Abstract base class for Rollout business processors.
Users should inherit this class and implement process() for custom agent
inference/rollout logic.
"""

from abc import abstractmethod

from dashscope.finetune.reinforcement.component.data.rollout_input import (
    RolloutInput,
)
from dashscope.finetune.reinforcement.component.data.rollout_output import (
    RolloutOutput,
)
from dashscope.finetune.reinforcement.component.processor.abstract_processor import (  # noqa: E501  # pylint: disable=line-too-long
    AbstractProcessor,
)


class AbstractRolloutProcessor(AbstractProcessor):
    """
    Abstract base class for Rollout business processors.

    Users must implement process() to define custom agent inference/rollout
    logic. Optionally override setup() to initialize workspace before the
    server starts.

    Example:
        >>> class MyRolloutProcessor(AbstractRolloutProcessor):
        ...     def setup(self) -> None:
        ...         # Load models, initialize connections, etc.
        ...         self.model = load_model("path/to/model")
        ...
        ...     def process(self, input: RolloutInput) -> RolloutOutput:
        ...         # Call model service for inference
        ...         response = self._call_model(input)
        ...         return RolloutOutput(
        ...             agent_output=AgentOutput(
        ...                 message=response.messages,
        ...             ),
        ...             status=TaskStatus.SUCCESS,
        ...         )
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
        requests are processed. The default implementation does nothing.

        Raises:
            Exception: If initialization fails, the server will fail to start.
        """

    @abstractmethod
    async def process(self, input_data: RolloutInput) -> RolloutOutput:
        """
        Execute rollout logic.

        Args:
            input_data: Parsed RolloutInput input object.

        Returns:
            RolloutOutput object containing standardized results (
            rollout_id, content, status).
        """
        raise NotImplementedError
