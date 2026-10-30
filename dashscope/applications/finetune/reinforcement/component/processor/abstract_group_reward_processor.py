# -*- coding: utf-8 -*-
"""
component/processor/abstract_group_reward_processor.py

Abstract base class for GroupReward business processors.
Users should inherit this class and implement process() for custom group
reward calculation logic.
"""

from abc import abstractmethod

from ..data import (
    GroupRewardInput,
)
from ..data import (
    GroupRewardOutput,
)
from .abstract_processor import (  # noqa: E501  # pylint: disable=line-too-long
    AbstractProcessor,
)


class AbstractGroupRewardProcessor(AbstractProcessor):
    """
    Abstract base class for GroupReward business processors.

    Users must implement process() to define custom group reward
    calculation logic. Optionally override setup() to initialize workspace
    before the server
    starts.

    Example:
        >>> class MyGroupRewardProcessor(AbstractGroupRewardProcessor):
        ...     def setup(self) -> None:
        ...         # Load embedding models, initialize databases, etc.
        ...         self.embedding_model = load_embedding_model()
        ...
        ...     def process(self, input: GroupRewardInput) ->
        GroupRewardOutput:
        ...         # Custom group reward calculation
        ...         rewards = []
        ...         for agent_output in input.agent_outputs:
        ...             score = self._compute_score(agent_output,
        input.ground_truth)
        ...             rewards.append(Reward(reward_score=score))
        ...         return GroupRewardOutput(
        ...             reward=GroupReward(rewards=rewards),
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
        requests are processed.
        The default implementation does nothing.

        Raises:
            Exception: If initialization fails, the server will fail to start.
        """

    @abstractmethod
    async def process(self, input_data: GroupRewardInput) -> GroupRewardOutput:
        """
        Execute group reward calculation logic.

        Args:
            input_data: Parsed GroupRewardInput input object.

        Returns:
            GroupRewardOutput object containing standardized results (
            rewards, status).
        """
        raise NotImplementedError
