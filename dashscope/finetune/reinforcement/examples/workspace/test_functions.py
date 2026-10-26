# -*- coding: utf-8 -*-
"""
Environment Variables Configuration:
--------------------------------------------------------------
1. DASHSCOPE_API_KEY (Required)
   - Sets the authentication key for DashScope API access
   - Format: API key string (e.g., "ds-1234567890abcdef")

2. LOG_LEVEL (Optional, default: 'info')
   - Controls logging verbosity
   - Allowed values: 'debug', 'info', 'warning', 'error', 'critical'

3. FC_PYPI_LIB (Required)
   - Specifies local/alternative dashscope package installations
   - Accepts space-separated list of wheel paths
   - Default: Uses official PyPI 'dashscope' package
   - Example: "dashscope-1.25.16-py3-none-any.whl"

Functions Workflow (main_functions):
--------------------------------------------------------------
register_functions():
- Registers and initializes function components for RL workflow
- Parameters:
  - functions: List of AgenticRLFunctionComponent configurations
  - lazy_load: If False, immediately loads components (default: False)
- Returns tuple of registered component IDs (rollout/reward/group_reward
entities and instances)

test_functions():
- Validates registered function components
- Parameters:
  - instance_id: Target component instance ID
  - type: FunctionType (ROLLOUT/REWARD/GROUP_REWARD)
  - input_data: Test input payload
- Verifies component outputs against expected results
"""

import json

from dashscope.finetune.reinforcement import (
    logger,
    TaskStatus,
    FunctionType,
    AgenticRLFunctionComponent,
    FunctionComponentModel,
)
from dashscope.finetune.agentic_rl import AgenticRL


async def main_functions():
    """Main execution workflow"""
    try:
        logger.info("Starting main tests functions")
        client = AgenticRL()

        functions = [
            # rollout-only
            AgenticRLFunctionComponent(
                type=FunctionType.ROLLOUT,
                fcmodel=FunctionComponentModel(
                    classpath="functions.rollout.rollout_only"
                    ".DemoRolloutProcessor",  # noqa: E501
                ),
            ),
            # reward
            AgenticRLFunctionComponent(
                type=FunctionType.REWARD,
                fcmodel=FunctionComponentModel(
                    classpath="functions.reward.reward.DemoRewardProcessor",
                ),
            ),
            # reward-decorator
            AgenticRLFunctionComponent(
                type=FunctionType.REWARD,
                fcmodel=FunctionComponentModel(
                    classpath="functions/reward/reward_decorator.py:SafetyProcessor",  # noqa: E501  # pylint: disable=line-too-long
                ),
            ),
            # group-reward
            AgenticRLFunctionComponent(
                type=FunctionType.GROUP_REWARD,
                fcmodel=FunctionComponentModel(
                    classpath="functions.reward.group_reward.DemoGroupRewardProcessor",  # noqa: E501  # pylint: disable=line-too-long
                ),
            ),
        ]

        # Register functions
        (
            rollout_entity_ids,
            reward_entity_ids,
            group_reward_entity_ids,
            rollout_instance_ids,
            reward_instance_ids,
            group_reward_instance_ids,
        ) = await client.register_functions(
            functions=functions,
            lazy_load=False,
        )  # register & load functions

        logger.info(
            f"agentic rl register functions: {rollout_entity_ids=},"
            f" {reward_entity_ids=}, {group_reward_entity_ids=},"
            f" {rollout_instance_ids=}, {reward_instance_ids=},"
            f" {group_reward_instance_ids=}",
        )

        with open(
            "./resources/rollout_input.json",
            "r",
            encoding="utf-8",
        ) as file:
            json_data = json.load(file)
            rollout_input = json_data

        with open(
            "./resources/reward_input.json",
            "r",
            encoding="utf-8",
        ) as file:
            json_data = json.load(file)
            reward_input = json_data

        with open(
            "./resources/reward_decorator_input.json",
            "r",
            encoding="utf-8",
        ) as file:
            json_data = json.load(file)
            reward_decorator_input = json_data

        with open(
            "./resources/group_reward_input.json",
            "r",
            encoding="utf-8",
        ) as file:
            json_data = json.load(file)
            group_reward_input = json_data

        # Testing rollout-only functions
        if rollout_instance_ids and rollout_instance_ids[0]:
            result = await AgenticRL.test_functions(
                instance_id=rollout_instance_ids[0],
                functype=FunctionType.ROLLOUT,
                input_data=rollout_input,
            )
            logger.info(
                f"agentic rl test rollout: {rollout_instance_ids[0]=},"
                f" {result=}",
            )
            status = result.get("status", None)
            assert (
                status == TaskStatus.SUCCESS
            ), f"Expected status, got {status}"

        # Testing reward functions
        if reward_instance_ids and reward_instance_ids[0]:
            result = await AgenticRL.test_functions(
                instance_id=reward_instance_ids[0],
                functype=FunctionType.REWARD,
                input_data=reward_input,
            )
            logger.info(
                f"agentic rl test rewards: {reward_instance_ids[0]=},"
                f" {result=}",
            )
            status = result.get("status", None)
            assert (
                status == TaskStatus.SUCCESS
            ), f"Expected status, got {status}"

        # Testing reward-decorator functions
        if (
            reward_instance_ids
            and len(
                reward_instance_ids,
            )
            > 1
            and reward_instance_ids[1]
        ):
            result = await AgenticRL.test_functions(
                instance_id=reward_instance_ids[1],
                functype=FunctionType.REWARD,
                input_data=reward_decorator_input,
            )
            logger.info(
                f"agentic rl test rewards-decorator: "
                f"{reward_instance_ids[1]=}, {result=}",
            )
            reward_score = result.get("reward", {}).get("reward_score", 0.0)
            assert (
                reward_score == 0.85
            ), f"Expected reward_score 0.85, got {reward_score}"

        # Testing group-reward functions
        if group_reward_instance_ids and group_reward_instance_ids[0]:
            result = await AgenticRL.test_functions(
                instance_id=group_reward_instance_ids[0],
                functype=FunctionType.GROUP_REWARD,
                input_data=group_reward_input,
            )
            logger.info(
                f"agentic rl test group-rewards: "
                f"{group_reward_instance_ids[0]=}, {result=}",
            )
            status = result.get("status", None)
            assert (
                status == TaskStatus.SUCCESS
            ), f"Expected status, got {status}"

        logger.info("All tests functions completed successfully")

    except Exception as e:
        root = e.root_cause if hasattr(e, "root_cause") else e
        logger.error(f"Main execution flow terminated: {root}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_functions())
