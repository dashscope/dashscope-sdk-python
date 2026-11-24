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

## Reward decorators

These three decorators target the **reward processor**, not rollout
function registration: `reward_func` is a class decorator applied to an
`AbstractRewardProcessor` subclass (it generates `process()` for you);
`sub_reward_func` and `aggregate_func` decorate processor methods to
declare sub-rewards and aggregation logic. Minimal example:

```python
from dashscope.applications.finetune.reinforcement import (
    AbstractRewardProcessor,
    RewardInput,
    RewardOutput,
    Reward,
    TaskStatus,
    reward_func,
    sub_reward_func,
)

@reward_func("MyReward")
class MyReward(AbstractRewardProcessor):
    @sub_reward_func("length", sub_weight=1.0)
    def length(self, input_data: RewardInput) -> RewardOutput:
        messages = input_data.agent_output.messages
        text = str(messages[0].get("content", "")) if messages else ""
        return RewardOutput(
            reward=Reward(reward_score=min(len(text) / 100, 1.0)),
            status=TaskStatus.SUCCESS,
        )
```

Without an `aggregate_func`, sub-rewards are combined as a weighted sum
using `sub_weight`. Rollouts are authored differently: subclass
`AbstractRolloutProcessor` and implement `process()` (optionally override
`setup()`); these decorators are not involved — see "Processor base
classes & demos" below.

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