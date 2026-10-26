# -*- coding: utf-8 -*-
"""
component/processor/abstract_reward_processor.py

Abstract base class for Reward business processors.
Users should inherit this class and implement process() for custom reward
calculation logic.
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, Executor
from typing import Optional, Dict

from dashscope.finetune.reinforcement.component.data.reward_input import (
    RewardInput,
)
from dashscope.finetune.reinforcement.component.data.reward_output import (
    RewardOutput,
)
from dashscope.finetune.reinforcement.component.func_decorator import (
    RewardProcessorMeta,
    SubRewardFunction,
    AggregateFunction,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractProcessor,
)


class AbstractRewardProcessor(AbstractProcessor):
    """
    Abstract base class for Reward business processors.

    Users must implement process() to define custom reward calculation logic.
    Optionally override setup() to initialize workspace before the server
    starts.

    Example:
        >>> class MyRewardProcessor(AbstractRewardProcessor):
        ...     def setup(self) -> None:
        ...         # Load embedding models, initialize databases, etc.
        ...         self.embedding_model = load_embedding_model()
        ...
        ...     def process(self, input: RewardInput) -> RewardOutput:
        ...         # Custom reward calculation
        ...         score = self._compute_score(input.agent_output,
        input.ground_truth)
        ...         return RewardOutput(
        ...             reward=Reward(
        ...                 reward_score=score,
        ...             ),
        ...             status=TaskStatus.SUCCESS,
        ...         )
    """

    reward_meta: RewardProcessorMeta
    functions_collected: bool = False

    def __init__(self, executor: Optional[Executor] = None):
        """
        Initialize reward processor

        Args:
            executor: Optional executor for running synchronous functions
        """
        self.executor = executor or ThreadPoolExecutor()
        self._executor_owned = executor is None
        # Collect all sub-functions and aggregate functions during
        # initialization
        # IMPORTANT: Subclasses MUST call super().__init__() to ensure
        # function collection
        self._collect_functions()

    def _collect_functions(self):
        """Collect marked sub-functions and aggregate functions in the class"""
        # Prevent duplicate collection
        if self.functions_collected:
            return

        # Ensure metadata exists
        if not hasattr(self, "reward_meta"):
            self.reward_meta = RewardProcessorMeta("default")

        # Iterate through class methods to collect marked functions
        # Only collect methods defined in this class (not from parent classes)
        for attr_name in dir(self.__class__):
            # Skip private and special attributes
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)

            # Skip non-method attributes
            if not callable(attr):
                continue

            # Check for sub-functions
            if hasattr(attr, "_is_sub_reward_func"):
                name = getattr(attr, "_sub_reward_name")
                weight = getattr(attr, "_sub_weight", 1.0)
                self.reward_meta.sub_functions[name] = SubRewardFunction(
                    name,
                    attr,
                    weight,
                )

            # Check for aggregate function
            if hasattr(attr, "is_aggregate_func"):
                self.reward_meta.aggregate_function = AggregateFunction(attr)

        self.functions_collected = True

    def shutdown(self):
        """Cleanup resources when the processor is no longer needed."""
        if self._executor_owned and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        """Ensure executor is cleaned up on garbage collection"""
        if (
            hasattr(self, "_executor_owned")
            and self._executor_owned
            and self.executor is not None
        ):
            self.executor.shutdown(wait=False)

    def get_weights(self):
        return {
            name: self.reward_meta.sub_functions[name].weight
            for name in self.reward_meta.sub_functions
        }

    def get_scores(self, sub_rewards: Dict[str, RewardOutput]):
        return {
            name: sub_rewards[name].reward.reward_score for name in sub_rewards
        }

    def get_total(self, sub_rewards: Dict[str, RewardOutput]):
        total = 0.0
        for name, reward_output in sub_rewards.items():
            if name not in self.reward_meta.sub_functions:
                raise ValueError(
                    f"Sub-reward function '{name}' is not registered. "
                    f"Available:"
                    f" {list(self.reward_meta.sub_functions.keys())}",
                )
            total += (
                reward_output.reward.reward_score
                * self.reward_meta.sub_functions[name].weight
            )
        return total

    def get_reward_metrics(self, sub_rewards: Dict[str, RewardOutput]):
        reward_metrics = {}
        for name, reward_output in sub_rewards.items():
            # Add namespace prefix to avoid key collisions
            for key, value in (
                reward_output.reward.reward_metrics or {}
            ).items():
                prefixed_key = f"{name}.{key}"
                reward_metrics[prefixed_key] = value

        return reward_metrics

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

    # pylint: disable=invalid-overridden-method
    @abstractmethod
    def process(self, input_data: RewardInput) -> RewardOutput:
        """
        Execute reward calculation logic.

        Args:
            input_data: Parsed RewardInput input object.

        Returns:
            RewardOutput object containing standardized results (score,
            rollout_id, status).
        """
        raise NotImplementedError
