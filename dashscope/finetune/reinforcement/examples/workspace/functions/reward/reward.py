# -*- coding: utf-8 -*-
"""
examples/workspace/reward.py

Demo implementation of a Reward Processor.
Demonstrates how to score Agent outputs based on simple rules.
"""

from __future__ import annotations

import math
import re
import string
import sympy

# Public names are provided lazily via __getattr__; __all__ entries are not
# bindings.
# pylint: disable=undefined-all-variable
from dashscope.finetune.reinforcement import RewardInput, RewardOutput
from dashscope.finetune.reinforcement import logger
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    TaskStatus,
)
from dashscope.finetune.reinforcement.component.data.reward_output import (
    Reward,
)

# pylint: disable=no-name-in-module
from dashscope.finetune.reinforcement.component.observability import (
    observe_processor,
)
from dashscope.finetune.reinforcement.component.processor import (
    AbstractRewardProcessor,
)


def normalize_option(option: str) -> str:
    """
    >>> normalize_option("  (A)  \n")
    'A'
    """
    return re.sub(r"(\s+|\(|\))", "", option)


def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


def float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def scalar_are_results_same(
    pred_result: str,
    true_result: str,
    rel_tol: float,
) -> bool:
    pred_result = (
        str(pred_result) if pred_result is not None else ""
    )  # type: ignore
    true_result = (
        str(true_result) if true_result is not None else ""
    )  # type: ignore

    if pred_result.strip() == true_result.strip():
        return True

    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result

    # The task is to calculate the result as a number
    try:
        pred_float = float_eval(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:
        pass

    return False


async def evaluate(prediction: str, ground_truth: str) -> float:
    match = re.search(
        r"###\s*ANSWER:\s*(.+?)(\s*###|$)",
        prediction,
        re.DOTALL,
    )
    answer_to_eval = match.group(1) if match else prediction
    return float(
        scalar_are_results_same(
            answer_to_eval,
            ground_truth,
            1e-2,
        ),
    )


class DemoRewardProcessor(AbstractRewardProcessor):
    def setup(self) -> None:
        """
        Initialize workspace before processing requests.

        Demo implementation: Logs startup message.
        In production, this could load embedding models, initialize
        databases, etc.
        """
        logger.info("[DemoRewardProcessor] setup() completed")

    # pylint: disable=invalid-overridden-method
    @observe_processor
    async def process(self, input_data: RewardInput) -> RewardOutput:
        """
        Demo implementation: Calculate simple rewards based on content length
        and ground_truth matching.

        Args:
            input_data: RewardInput input parameter.

        Returns:
            RewardOutput object containing scoring results.
        """
        logger.info("[DemoRewardProcessor] computing reward for rollout_id")
        messages = input_data.agent_output.messages
        content = messages[-1].get("content", "") if messages else ""
        score = await evaluate(
            str(content or ""),
            str(input_data.ground_truth or ""),
        )

        result = RewardOutput(
            reward=Reward(
                reward_score=score,
                reward_metrics={"test1": 0.5, "test2": 0.3},
            ),
            status=TaskStatus.SUCCESS,
            error=None,
        )
        logger.info(f"[DemoRewardProcessor2][sync] result: {result}")
        return result
