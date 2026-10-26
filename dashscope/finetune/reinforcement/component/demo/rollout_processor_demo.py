# -*- coding: utf-8 -*-
"""
component/demo/rollout_processor_demo.py

Demo implementation of Rollout Processor.
Demonstrates simple rollout execution flow (simulating Agent invocation).
"""

import time

from dashscope.finetune.reinforcement.common.log import logger
from dashscope.finetune.reinforcement.component.data.base_data_model import (
    AgentOutput,
    TaskStatus,
)
from dashscope.finetune.reinforcement.component.data.rollout_input import (
    RolloutInput,
)
from dashscope.finetune.reinforcement.component.data.rollout_output import (
    RolloutOutput,
)
from dashscope.finetune.reinforcement.component.processor.abstract_rollout_processor import (  # noqa: E501  # pylint: disable=line-too-long
    AbstractRolloutProcessor,
)


class DemoRolloutProcessor(AbstractRolloutProcessor):
    """
    Demo implementation of Rollout Processor.
    Demonstrates simple rollout execution flow (simulating Agent invocation).

    In production, this should invoke real Agent/model services,
    e.g., through OpenAI SDK or Dashscope SDK.
    """

    def setup(self) -> None:
        """
        Initialize workspace before processing requests.

        Demo implementation: Logs startup message.
        In production, this could load ML models, establish connections, etc.
        """
        logger.info(
            "[DemoRolloutProcessor] setup() called - initializing workspace",
        )
        # Demo: No actual initialization needed
        # In production, you might:
        # - Load ML models into memory
        # - Initialize API clients
        # - Set up connection pools
        logger.info("[DemoRolloutProcessor] setup() completed")

    # pylint: disable=invalid-overridden-method
    def process(self, input_data: RolloutInput) -> RolloutOutput:
        """
        Demo implementation: Simulates Agent invocation by echoing messages.

        Args:
            input_data: RolloutInput input parameter

        Returns:
            RolloutOutput object containing standardized execution results
        """
        logger.info(
            f"[DemoRolloutProcessor] starting rollout | "
            f"model={input_data.model_resource.model_name}",
        )

        start = time.time()

        # Demo: Return messages as echo content
        messages = list(input_data.messages)

        # Add a simple assistant response
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        assistant_content = f"[DemoRolloutProcessor Echo] {last_user_msg}"
        messages.append({"role": "assistant", "content": assistant_content})

        latency = round(time.time() - start, 4)

        result = RolloutOutput(
            agent_output=AgentOutput(
                messages=messages,
                rollout_extra=input_data.rollout_extra,
                rollout_metrics={},
            ),
            status=TaskStatus.SUCCESS,
            error=None,
        )
        logger.info(f"[DemoRolloutProcessor] result: latency={latency}s")
        return result
