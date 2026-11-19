---
description: "Agentic RL 微调 API 参考：提交入口与 reward/rollout 编写 API。"
---
# 智能体强化学习微调（Agentic RL）

`dashscope.applications.finetune` 承载 Agentic RL 微调流程：
用你自己编写的 rollout / reward 函数提交训练任务。服务端的任务、
数据集与函数管理在 `client.rl` 下（见
[资源](resources.md#reinforcement-learning)）；本页覆盖本地提交入口
与 reward / rollout 组件的编写 API。

## 提交入口

::: dashscope.applications.finetune.agentic_rl.AgenticRL

::: dashscope.applications.finetune.agentic_rl.SubmitOutput

::: dashscope.applications.finetune.agentic_rl.SubmitResult

## Reward 装饰器

这三个装饰器面向 **reward processor**，不是 rollout 函数的注册接口：
`reward_func` 是类装饰器，挂在 `AbstractRewardProcessor` 子类上并
自动生成 `process()`；`sub_reward_func` 与 `aggregate_func` 装饰
processor 的方法，分别声明子奖励与聚合逻辑。最小示例：

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

不写 `aggregate_func` 时，默认按 `sub_weight` 加权求和。
rollout 的编写方式不同：继承 `AbstractRolloutProcessor` 并实现
`process()`（可选覆写 `setup()`），不使用这些装饰器，
见下方「处理器基类与示例」。

::: dashscope.applications.finetune.reinforcement.component.reward_func

::: dashscope.applications.finetune.reinforcement.component.sub_reward_func

::: dashscope.applications.finetune.reinforcement.component.aggregate_func

## 数据模型

::: dashscope.applications.finetune.reinforcement.component.RolloutInput

::: dashscope.applications.finetune.reinforcement.component.RolloutOutput

::: dashscope.applications.finetune.reinforcement.component.RewardInput

::: dashscope.applications.finetune.reinforcement.component.Reward

::: dashscope.applications.finetune.reinforcement.component.RewardOutput

::: dashscope.applications.finetune.reinforcement.component.AgentOutput

::: dashscope.applications.finetune.reinforcement.component.Task

::: dashscope.applications.finetune.reinforcement.component.TaskStatus

## 处理器基类与示例

::: dashscope.applications.finetune.reinforcement.component.AbstractRolloutProcessor

::: dashscope.applications.finetune.reinforcement.component.AbstractRewardProcessor

::: dashscope.applications.finetune.reinforcement.component.DemoRolloutProcessor

::: dashscope.applications.finetune.reinforcement.component.DemoRewardProcessor