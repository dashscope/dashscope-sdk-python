# -*- coding: utf-8 -*-
"""
examples/workspace/reward.py

Demo implementation of a Reward Processor.
Demonstrates how to score Agent outputs based on simple rules.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict

from dashscope.finetune.reinforcement import AbstractRewardProcessor
from dashscope.finetune.reinforcement import (
    RewardInput,
    RewardOutput,
    Reward,
    TaskStatus,
)
from dashscope.finetune.reinforcement import logger
from dashscope.finetune.reinforcement import (
    reward_func,
    sub_reward_func,
    aggregate_func,
)


@reward_func("SafetyProcessor")
class SafetyProcessor(AbstractRewardProcessor):
    BANNED = {"hack", "exploit", "attack"}

    @sub_reward_func("toxicity", sub_weight=0.7)
    def toxicity(self, input_data: RewardInput) -> RewardOutput:
        logger.info("[SafetyProcessor][Sync] toxicity ...")
        time.sleep(2)

        messages = input_data.agent_output.messages
        content = messages[0].get("content", "") if messages else ""
        response = str(content) if content is not None else ""
        has_banned = any(w in response.lower() for w in self.BANNED)
        logger.info("[SafetyProcessor][Sync] toxicity end!!!")

        user_reward_metrics = {"score-1": 0.1}
        return RewardOutput(
            reward=Reward(
                reward_score=0.0 if has_banned else 1.0,
                reward_metrics=user_reward_metrics,
            ),
            status=TaskStatus.SUCCESS,
            error=None,
        )

    @sub_reward_func("refusal", sub_weight=0.3)
    async def refusal(self, input_data: RewardInput) -> RewardOutput:
        logger.info("[SafetyProcessor][Async] refusal ...")
        await asyncio.sleep(3)

        messages = input_data.agent_output.messages
        content = messages[0].get("content", "") if messages else ""
        response = str(content) if content is not None else ""
        is_refusal = response.strip().lower().startswith("i cannot")
        logger.info("[SafetyProcessor][Async] refusal end!!!")

        user_reward_metrics = {"score-1": 0.2}
        return RewardOutput(
            reward=Reward(
                reward_score=0.5 if is_refusal else 1.0,
                reward_metrics=user_reward_metrics,
            ),
            status=TaskStatus.SUCCESS,
            error=None,
        )

    @aggregate_func
    async def aggregate(
        self,
        sub_rewards: Dict[str, RewardOutput],
    ) -> RewardOutput:
        logger.info("[SafetyProcessor][Async] computing reward for rollout_id")

        weights = self.get_weights()
        scores = self.get_scores(sub_rewards)
        if len(scores) != len(weights):
            raise ValueError("scores and weights must have the same length")

        # 0.5 * 0.3 + 1.0 * 0.7 = 0.85
        total = sum(scores[k] * weights[k] for k in scores)
        logger.info(f"[SafetyProcessor][Async] result: {total}")

        # user reward_metrics
        reward_metrics = self.get_reward_metrics(sub_rewards)
        user_reward_metrics = {"aggregate-1": 0.4}
        user_reward_metrics.update(reward_metrics)
        return RewardOutput(
            reward=Reward(
                reward_score=total,
                reward_metrics=user_reward_metrics,
            ),
            status=TaskStatus.SUCCESS,
            error=None,
        )
