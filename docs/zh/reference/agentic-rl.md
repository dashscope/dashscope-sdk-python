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

## 函数装饰器

把普通 Python 函数注册为 rollout / reward 组件：

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