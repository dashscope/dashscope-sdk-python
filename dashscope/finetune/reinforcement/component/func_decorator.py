# -*- coding: utf-8 -*-
import asyncio
import copy
from functools import wraps
from typing import Callable, Dict, Optional, Type

from dashscope.finetune.reinforcement.common.log import logger
from dashscope.finetune.reinforcement.component.data import (
    RewardInput,
    RewardOutput,
    Reward,
    TaskStatus,
)


class SubRewardFunction:
    def __init__(self, name: str, func: Callable, weight: float = 1.0):
        self.name = name
        self.func = func
        self.weight = weight
        self.score = 0.0
        self.reward_metrics = {}

    def __deepcopy__(self, memo):
        """Create a deep copy of SubRewardFunction, resetting mutable state"""
        # Functions are not copied - they're considered immutable
        return SubRewardFunction(
            name=copy.deepcopy(self.name, memo),
            func=self.func,  # Keep the same function reference
            weight=copy.deepcopy(self.weight, memo),
        )


class AggregateFunction:
    def __init__(self, func: Callable):
        self.func = func

    def __deepcopy__(self, memo):
        """Create a deep copy of AggregateFunction"""
        return AggregateFunction(
            func=self.func,  # Keep the same function reference
        )


class RewardProcessorMeta:
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.sub_functions: Dict[str, SubRewardFunction] = {}
        self.aggregate_function: Optional[AggregateFunction] = None

        self._copy_cache = None

    def __deepcopy__(self, memo):
        """Create a deep copy of RewardProcessorMeta"""
        new_meta = RewardProcessorMeta(
            processor_id=copy.deepcopy(self.processor_id, memo),
        )

        # Deep copy all sub-functions
        new_meta.sub_functions = {
            name: copy.deepcopy(sub_func, memo)
            for name, sub_func in self.sub_functions.items()
        }

        # Deep copy aggregate function if exists
        if self.aggregate_function:
            new_meta.aggregate_function = copy.deepcopy(
                self.aggregate_function,
                memo,
            )

        return new_meta

    def copy(self):
        """Cached copy method to avoid repeated deepcopies"""
        if not self._copy_cache:
            self._copy_cache = copy.deepcopy(self)
        return self._copy_cache


# ================= Decorator Implementation =================
def reward_func(processor_id: str) -> Callable[[Type], Type]:
    """Class decorator to mark reward processors and collect sub-function
    information"""

    def decorator(cls: Type) -> Type:
        # Create metadata object and attach to class
        meta = RewardProcessorMeta(processor_id)
        setattr(cls, "reward_meta", meta)

        async def process(self, input_data: RewardInput) -> RewardOutput:
            """Processes input by executing all sub-reward functions and
            aggregating results"""
            sub_rewards: Dict[str, RewardOutput] = {}

            tasks = []
            for name, sub_func in self.reward_meta.sub_functions.items():
                # Use the decorated function directly
                func = sub_func.func

                # Check if it's a coroutine function
                if asyncio.iscoroutinefunction(func):
                    task = func(input_data)
                else:
                    # Run synchronous function in executor
                    loop = asyncio.get_running_loop()
                    task = loop.run_in_executor(
                        self.executor,
                        func,
                        input_data,
                    )
                tasks.append((name, task))

            # Wait for all tasks to complete
            for name, task in tasks:
                try:
                    result = await task
                    sub_rewards[name] = result

                except Exception as e:
                    logger.error(
                        f"Error in sub-reward function {name}: {str(e)}",
                    )
                    # Store error as a zero score
                    sub_rewards[name] = RewardOutput(
                        reward=Reward(reward_score=0.0, reward_metrics={}),
                        status=TaskStatus.FAILED,
                        error=str(e),
                    )

            # Call aggregation function if available
            if self.reward_meta.aggregate_function:
                # Get the underlying function (unbound method)
                func = self.reward_meta.aggregate_function.func

                if asyncio.iscoroutinefunction(func):
                    return await func(sub_rewards)
                else:
                    # Run synchronous aggregate function in executor
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        self.executor,
                        func,
                        sub_rewards,
                    )

            # If no aggregate function, use default weighted sum
            total = self.get_total(sub_rewards)

            # Return RewardOutput with calculated total
            return RewardOutput(
                reward=Reward(
                    reward_score=total,
                ),
                status=TaskStatus.SUCCESS,
                error=None,
            )

        # Set the new process method
        setattr(cls, "process", process)

        # Fix abstract method issue
        if hasattr(cls, "__abstractmethods__"):
            # Create a new set without 'process'
            abstract_methods = set(cls.__abstractmethods__)
            abstract_methods.discard("process")
            cls.__abstractmethods__ = frozenset(abstract_methods)

        return cls

    return decorator


def sub_reward_func(
    name: Optional[str] = None,
    sub_weight: float = 1.0,
) -> Callable[[Callable], Callable]:
    """Decorator to mark sub-reward functions and specify weights"""

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        # Preserve async nature of the original function
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self, input_data: RewardInput):
                return await func(self, input_data)

            wrapper = async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(self, input_data: RewardInput):
                return func(self, input_data)

            wrapper = sync_wrapper

        # Attach metadata
        setattr(wrapper, "_is_sub_reward_func", True)
        setattr(wrapper, "_sub_reward_name", func_name)
        setattr(wrapper, "_sub_weight", sub_weight)
        return wrapper

    return decorator


def aggregate_func(func: Callable) -> Callable:
    """Decorator to mark aggregation functions"""
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)

        wrapper = async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper = sync_wrapper

    wrapper.is_aggregate_func = True
    return wrapper
