---
description: "API reference for Agentic RL fine-tuning: submission entry point and reward/rollout authoring API."
---
# Agentic RL Fine-tuning

`dashscope.applications.finetune` hosts the Agentic RL fine-tuning workflow:
submit training jobs built from your own rollout / reward functions. Job,
dataset and function management on the service side lives under `client.rl`
(see [Resources](resources.md#reinforcement-learning)); this page covers the
local submission entry point and the authoring API for reward / rollout
components.

## Submission entry point

::: dashscope.applications.finetune.agentic_rl.AgenticRL

::: dashscope.applications.finetune.agentic_rl.SubmitOutput

::: dashscope.applications.finetune.agentic_rl.SubmitResult

## Function decorators

Register plain Python functions as rollout / reward components:

::: dashscope.applications.finetune.reinforcement.component.reward_func

::: dashscope.applications.finetune.reinforcement.component.sub_reward_func

::: dashscope.applications.finetune.reinforcement.component.aggregate_func

## Data models

::: dashscope.applications.finetune.reinforcement.component.RolloutInput

::: dashscope.applications.finetune.reinforcement.component.RolloutOutput

::: dashscope.applications.finetune.reinforcement.component.RewardInput

::: dashscope.applications.finetune.reinforcement.component.Reward

::: dashscope.applications.finetune.reinforcement.component.RewardOutput

::: dashscope.applications.finetune.reinforcement.component.AgentOutput

::: dashscope.applications.finetune.reinforcement.component.Task

::: dashscope.applications.finetune.reinforcement.component.TaskStatus

## Processor base classes & demos

::: dashscope.applications.finetune.reinforcement.component.AbstractRolloutProcessor

::: dashscope.applications.finetune.reinforcement.component.AbstractRewardProcessor

::: dashscope.applications.finetune.reinforcement.component.DemoRolloutProcessor

::: dashscope.applications.finetune.reinforcement.component.DemoRewardProcessor