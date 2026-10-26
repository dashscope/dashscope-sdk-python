# Agentic Reinforcement Learning: 用户指南 [[English]](./UserGuide.md)

---

## 1. 简介

**Agentic RL SDK/CLI** 提供了一套完整的工具链，用于为大语言模型（LLMs）构建、训练和管理强化学习（RL）模型。它简化了定义智能体行为、收集轨迹和优化策略的复杂工作流。

SDK 包含两个核心模块：
1.  **函数模块**：管理用于 **Rollout**（轨迹生成）、**Reward**（评分）和 **Group Reward**（批量评分）的自定义 Python 代码。支持自动注册、测试和内置**可观测性（Tracing）**。
2.  **调优模块**：处理数据集管理、超参数配置、作业提交和生命周期管理（状态、日志、取消）。

**工作流**：
1. 注册自定义函数。普通强化学习必须配置 Rollout 和至少一个 Reward；
   OPD 的 Rollout、Reward 可选。
2. 使用调优模块上传数据集、配置作业并提交训练。

---

## 2. 安装与设置

### 2.1 通过 PyPI 安装
```bash
pip install dashscope>=1.25.19
```

### 2.2 从源码安装（开发环境）
```bash
git clone https://github.com/dashscope/dashscope-sdk-python.git
cd dashscope-sdk-python
pip install -e .  # 以可编辑模式安装，便于开发
```

### 2.3 授权认证
1.  获取您的 **DashScope API Key**。
2.  设置环境变量：
    ```bash
    export DASHSCOPE_API_KEY='your-api-key-here'
    ```

### 2.4 项目结构
推荐的工作空间结构确保平滑部署和本地测试：

```text
workspace/
├── data/                   # 数据集
│   ├── calc_train_min.jsonl
│   └── calc_validation_min.jsonl
├── functions/              # 自定义函数；仅 OPD 可省略 Rollout/Reward
│   ├── reward/
│   │   ├── group_reward_.py
│   │   ├── reward.py
│   │   └── reward_decorator.py
│   └── rollout/
│       ├── rollout_only.py
│       └── rollout.py
├── requirements.txt        # 函数组件依赖项
├── rl-job.yaml             # 普通 Agentic RL 配置
└── opd-job.yaml            # OPD 配置
```

普通强化学习必须配置自定义 Rollout 和至少一个 Reward。只有 OPD 可以不配置
Rollout、Reward，或者按需只配置其中一个。

### 2.5 依赖包（requirements.txt 指南）
此文件对于将函数组件部署到云端是**必需的**，必须位于工作空间根目录。

**requirements.txt:**
```txt
fastapi==0.136.0
uvicorn==0.45.0
typer==0.24.1
rich==15.0.0
pyyaml==6.0.3
protobuf>=4.25.8,<7.0 #6.33.6
fsspec==2026.3.0
httpx==0.28.1
tenacity==9.1.4
...
```

**关键注意事项:**
*   **默认包**：运行时环境中已预装 `dashscope`，请**不要**将其包含在您的 `requirements.txt` 中。
*   **Protobuf**：必须在指定版本范围内以避免兼容性问题。

#### 可观测性（Tracing）依赖项
为使用 **Tracing（链路跟踪）** 能力（处理器 / LLM / 工具调用链路），请在您的 `requirements.txt` 中添加以下依赖（推荐固定版本以保证部署可复现）：

```txt
opentelemetry-api==1.41.1
opentelemetry-sdk==1.41.1
opentelemetry-exporter-otlp-proto-http==1.41.1
opentelemetry-processor-baggage==0.62b1
loongsuite-util-genai==0.4.0
```

### 2.6 日志配置
通过 `LOG_LEVEL` 环境变量配置日志详细程度：

```bash
export LOG_LEVEL="DEBUG"   # 最详细（会屏蔽敏感信息如 API 密钥）
export LOG_LEVEL="INFO"    # 默认
export LOG_LEVEL="WARNING"
export LOG_LEVEL="ERROR"
export LOG_LEVEL="CRITICAL"
```

---

## 3. 编写函数
参考：workspace/functions 目录，输出 rollout.py、reward.py 等文件

### 3.1 实现函数

函数必须继承 SDK 提供的抽象基类。

#### Rollout 处理器
生成智能体轨迹（与环境/LLM 的交互）。

```python
from dashscope.finetune.reinforcement import AbstractRolloutProcessor, RolloutInput, RolloutOutput

class DemoRolloutProcessor(AbstractRolloutProcessor):
    async def process(self, input: RolloutInput) -> RolloutOutput:
        # 生成轨迹（支持 async/def）
        pass
```

#### Reward 处理器
评分单个步骤或最终输出。

```python
from dashscope.finetune.reinforcement import AbstractRewardProcessor, RewardInput, RewardOutput

class DemoRewardProcessor(AbstractRewardProcessor):
    def process(self, input: RewardInput) -> RewardOutput:
        # 计算奖励分数
        pass
```

#### 使用装饰器的高级 Reward 处理器
```python
from dashscope.finetune.reinforcement import reward_func, sub_reward_func, aggregate_func

@reward_func("SafetyProcessor")
class SafetyProcessor(AbstractRewardProcessor):
    @sub_reward_func("toxicity", sub_weight=0.7)
    def toxicity(self, input: RewardInput) -> RewardOutput: ...

    @sub_reward_func("refusal", sub_weight=0.3)
    async def refusal(self, input: RewardInput) -> RewardOutput: ...

    @aggregate_func
    async def aggregate(self, sub_rewards: dict[str, RewardOutput]) -> RewardOutput: # 自定义聚合逻辑
        weights = self.get_weights()
        scores = self.get_scores(sub_rewards)
        reward_metrics = self.get_reward_metrics(sub_rewards)

        total = ...  # 计算总奖励
        return RewardOutput(...)
```

#### Group Reward 处理器
批量评分轨迹（例如用于排名）。

```python
from dashscope.finetune.reinforcement import AbstractGroupRewardProcessor, GroupRewardInput, GroupRewardOutput

class DemoGroupRewardProcessor(AbstractGroupRewardProcessor):
    def setup(self) -> None:
        pass

    async def process(self, input: GroupRewardInput) -> GroupRewardOutput:
        # 计算组奖励
        pass
```

### 3.2 可观测性（Tracing）

使用 OpenTelemetry 实现对智能体执行的深度可见性。在控制台完成 ARMS 授权后，链路跟踪数据会自动导出到 **ARMS**（阿里云实时监控服务）；您可以在链路中查看一次 Rollout 内的 LLM 与工具调用等节点。参见 [ARMS 文档](https://help.aliyun.com/zh/arms/?spm=5176.30275541.J_ZGek9Blx07Hclc3Ddt9dg.3.3ce02f3dmKOpPK&scm=20140722.S_card@@%E4%BA%A7%E5%93%81@@596792.S_new~UND~card.ID_card@@%E4%BA%A7%E5%93%81@@596792-RL_arms-LOC_2024SPSearchCard-OR_ser-PAR1_0bc1409817757870159831522e3953-V_4-RE_new5-P0_0-P1_0)。

#### 先决条件
1.  将可观测依赖项添加到 `requirements.txt`（见第 2.5 节"可观测性（Tracing）依赖项"）。
2.  确认未显式关闭链路跟踪（`ENABLE_TRAJECTORY=false` 会关闭）。

#### 检测装饰器
从 `dashscope.finetune.reinforcement.component.observability` 导入。

| 装饰器 | 用法 | 描述 |
| :--- | :--- | :--- |
| `@observe_processor` | 在 `process()` 方法上 | 自动跟踪输入/输出、延迟和状态。自动确定 span 类型（ROLLOUT/REWARD）。 |
| `trace_client()` | 在 `setup()` 中或在首次 LLM 调用之前 | 包装 LLM 客户端（OpenAI、DashScope、LangChain 类）。自动跟踪所有下游调用。**推荐使用**。 |
| `@observe_llm` | 在自定义 LLM 函数上 | 如果 `trace_client` 不支持您的包装器时使用。需要 `model` 和 `messages` 作为关键字参数。 |
| `trace_tool()` | 在 `setup()` 中（工具创建后） | 包装工具（LangChain/MCP/LangGraph 类）。自动跟踪工具调用。 |
| `@observe_tool` | 在普通函数上 | 用于未包装为 BaseTools 的简单 Python 函数。 |

> **Setup:** 在服务器启动时调用一次；支持同步和异步。同步 setup 会被卸载以避免阻塞事件循环。

> **快速提示：** `@observe_llm` 适合用于自定义函数：`@observe_llm async def call_llm(*, model, messages, **kwargs): ...`；`@observe_tool` 适合用于普通函数工具：`@observe_tool def my_tool(x: str) -> str: ...`。

#### `trace_client()` 支持的内容（鸭子类型）
`trace_client(client)` 通过结构检测（非类名）。它支持：

- **完整的 OpenAI 客户端** 公开 `.chat.completions.create`
- **Completions 资源** 公开 `.create` 但**没有** `.chat`（例如 `ChatOpenAI.client`）
- **LangChain 类包装器** 公开 `.client` 和/或 `.async_client`
- **DashScope Generation** 类（传递类本身，该类具有 `call` 类方法）

> **注意：** 若未产生链路跟踪数据，可能是客户端类型不在支持范围内；此时会跳过跟踪但不影响业务执行。
> **DashScope 最小用法：** 对 `Generation` 类调用 `trace_client(Generation)`（传入类，而非实例）。

#### `trace_tool()` 支持的内容
`trace_tool(tools)` 接受以下形式：

- 单个工具对象（例如 LangChain `BaseTool`）
- 工具列表/元组
- 名称到工具的映射字典
- LangGraph `ToolNode`（工具在内部扩展）
- `langchain-mcp-adapters` 返回的 MCP 工具（提供者自动设置为 `"mcp"`）

> **MCP 注意：** MCP 服务器和客户端在独立进程中运行。在服务器端函数上使用 `@observe_tool` 对客户端没有影响。在客户端调用 `get_tools()` 后始终调用 `trace_tool(tools)`。

#### 示例：检测的 Rollout 处理器

> 以下示例主要演示 Tracing 接入方式，非完整对话状态机或业务逻辑示例。

```python
import openai
from dashscope.finetune.reinforcement import AbstractRolloutProcessor, RolloutInput, RolloutOutput
from dashscope.finetune.reinforcement.component.data.base_data_model import AgentOutput, TaskStatus
from dashscope.finetune.reinforcement.component.observability import (
    observe_processor,
    trace_client,
    trace_tool,
)

class MyRolloutProcessor(AbstractRolloutProcessor):

    async def setup(self) -> None:
        # 1. 跟踪 LLM 客户端
        self._client = openai.AsyncOpenAI(base_url="...", api_key="...")
        trace_client(self._client)

        # 2. 跟踪工具（例如 MCP）
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({...})
        self._tools = await client.get_tools()
        trace_tool(self._tools)

    @observe_processor
    async def process(self, input: RolloutInput) -> RolloutOutput:
        messages = input.messages or []
        model = input.model_resource.model_name

        # 由于 trace_client(self._client)，此调用会自动被跟踪
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = response.choices[0].message.content if response.choices else ""

        return RolloutOutput(
            agent_output=AgentOutput(messages=messages, reward_score=0.0),
            status=TaskStatus.SUCCESS,
        )
```

> **提示：** DashScope 的部分流式（`stream`）场景可能不生成链路跟踪数据；排查时可先改用非流式验证。

---

## 4. SDK & CLI 参考
* [SDK] 接口类: dashscope.finetune.agentic_rl.AgenticRL
* [CLI] 入口点: dashscope rl

### 4.1 作业配置

普通强化学习和 OPD 使用同一套 SDK/CLI 工作流，区别在于传入的 YAML 配置。
代码参数会覆盖 YAML 配置。

| 训练方式 | 配置文件 | 自定义函数 |
|---|---|---|
| 普通强化学习 | `rl-job.yaml` | Rollout 必选，Reward 至少一个 |
| OPD | `opd-job.yaml` | Rollout、Reward 独立可选，可以都不配置 |

**[SDK] \_\_init\_\_**
```python
def __init__(self, api_key: str = None): ...
```
初始化 AgenticRL 实例。

**参数**:
- `api_key`: 认证用的 API 密钥（如未提供则使用环境变量）

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
rl = AgenticRL(api_key="your_api_key")
```

**[SDK] [init](submit_job.py)**
```python
def init(self, config_path: Optional[str] = None, **kwargs) -> Self: ...
```
从 YAML 配置文件初始化实例。

**参数**:
- `config_path`: YAML 配置文件路径
- `**kwargs`: 配置覆盖项

**返回**: 自身实例

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
config_path = "opd-job.yaml"  # 普通强化学习使用 "rl-job.yaml"
rl = AgenticRL().init(config_path, job_name="custom_job")
result = await rl.run()
```

选择 `opd-job.yaml` 时，可以通过保留或删除 Rollout、Reward 函数块选择
需要的能力：

| 保留的函数块 | OPD 场景 | 轨迹来源 | 任务奖励 |
|---|---|---|---|
| 都不保留 | 仅 Teacher | 平台生成 | 不启用 |
| 仅 Reward | Teacher + Reward | 平台生成 | 自定义 Reward |
| 仅 Rollout | Teacher + Rollout | 自定义 Rollout | 不启用 |
| Rollout 和 Reward | Teacher + Rollout + Reward | 自定义 Rollout | 自定义 Reward |

仓库提交的 `opd-job.yaml` 默认保留两个函数块。删除其中一个或两个，即可
切换为其他 OPD 组合：

```yaml
teacher_model: qwen3.5-397b-a17b

functions:
# 删除此块后使用平台生成。
- type: rollout
  # ...
# 删除此块后不使用自定义任务奖励。
- type: reward
  # ...

training:
  type: pg_opd
```

```bash
# 普通强化学习
dashscope rl run -c rl-job.yaml

# OPD
dashscope rl run -c opd-job.yaml
```

### 4.2 注册函数

上传代码并注册函数组件（FCs）。

**[SDK] [register_functions](submit_job.py)**
```python
def register_functions(self, functions: Optional[Union[List[Union[RolloutFunctionComponent, RewardFunctionComponent]], RolloutFunctionComponent, RewardFunctionComponent]] = None, lazy_load: Optional[bool] = True) -> tuple: ...
```
注册函数组件。

**参数**:
- `functions`: 要注册的函数组件
- `lazy_load`: 延迟加载直到执行

**返回**: 实体/实例 ID 元组

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL, AgenticRLFunctionComponent, FunctionType, FunctionComponentModel

rl=AgenticRL()
rollout_eids, reward_eids, group_eids, \
rollout_iids, reward_iids, group_iids = await rl.register_functions(
    functions=[
        AgenticRLFunctionComponent(
            type=FunctionType.ROLLOUT,
            fcmodel=FunctionComponentModel(
                zipdir='./',
                classpath="functions.rollout.demo_rollout2.DemoRolloutProcessor"),
        ),

        AgenticRLFunctionComponent(
            type=FunctionType.REWARD,
            fcmodel=FunctionComponentModel(
                classpath="functions/reward/demo_reward.py:DemoRewardProcessor"),
        ),

        AgenticRLFunctionComponent(
            type=FunctionType.GROUP_REWARD,
            fcmodel=FunctionComponentModel(
                classpath="functions.reward.demo_group_reward.DemoGroupRewardProcessor"),
        ),
    ],
    lazy_load=False  # 设为 False 可立即获取实例 ID 用于测试
)
```

**[CLI] register_functions**

**用法: dashscope register_functions [OPTIONS]**
```bash
 🧩 注册 Rollout/Reward 函数组件，返回 entity_id & instance_id

 至少需要以下一项:
 - rollout_classpath
 - reward_classpaths

╭─ 选项 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --rollout-classpaths                             TEXT  Rollout 类路径列表 (file.py:ClassName)                                                                                                               │
│ --reward-classpaths                              TEXT  Reward 类路径列表 (file.py:ClassName)                                                                                                                │
│ --group-reward-classpaths                        TEXT  Group-reward 类路径列表 (file.py:ClassName)                                                                                                          │
│ --workspace-dir                                  TEXT  本地工作空间目录 [默认: ./]                                                                                                                       │
│ --lazy-load                    --no-lazy-load          延迟实例加载（设为 False 用于调试） [默认: lazy-load]                                                                                         │
│ --api-key                                        TEXT  DashScope API Key（如省略则使用 DASHSCOPE_API_KEY 环境变量） [环境变量: DASHSCOPE_API_KEY]                                                                    │
│ --output-format            -o                    TEXT  输出格式: table|json|yaml [默认: json]                                                                                                                │
│ --help                                                 显示此消息并退出                                                                                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
**示例**:
```bash
dashscope rl register_functions \
  --rollout-classpath "functions.rollout.rollout_only.DemoRolloutProcessor" \
  --group-reward-classpaths "functions.reward.group_reward.DemoGroupRewardProcessor" \
  --workspace-dir "./" \
  --output-format json
```

### 4.3 测试函数

#### 4.3.1 远程测试
**[SDK] [test_functions](test_functions.py)**

使用样本数据测试已注册实例。

```python
def test_functions(cls, instance_id: str, type: FunctionType, input_data: Dict[str, Any], api_key: str = None): ...
```

**参数**:
- `instance_id`: 函数实例 ID
- `type`: 函数类型 (ROLLOUT/REWARD/GROUP_REWARD)
- `input_data`: 测试输入数据
- `api_key`: 认证用的 API 密钥

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL, FunctionType

# 测试 Rollout
result = await AgenticRL.test_functions(
    instance_id=rollout_iids[0],
    functype=FunctionType.ROLLOUT,
    input_data="resouces/rollout_input.json" # JSON 文件路径
)

# 测试 Reward
reward_input = {
    "func_type": "reward",
    "agent_output": {
        "messages": [{"role": "user", "content": "Test"}],
        "reward_score": null
    }
}
result = await AgenticRL.test_functions(
    instance_id=reward_iids[0],
    functype=FunctionType.REWARD,
    input_data=reward_input
)
```

**[CLI] test_functions**

**用法: dashscope test_functions [OPTIONS] INSTANCE_ID**
```bash
 🧪 使用自定义输入数据测试已注册的 Rollout/Reward 函数实例

╭─ 参数 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    instance_id      TEXT  目标函数实例 ID (例如 ro-ins-xxx 或 rw-ins-xxx) [必需]                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 选项 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --type           -t      TEXT  函数类型: ROLLOUT 或 REWARD [必需]                                                                                                                                        │
│ *  --input          -i      TEXT  包含测试负载的 JSON 字符串或文件路径 [必需]                                                                                                                        │
│    --api-key                TEXT  DashScope API Key（如省略则使用 DASHSCOPE_API_KEY 环境变量） [环境变量: DASHSCOPE_API_KEY]                                                                                         │
│    --output-format  -o      TEXT  输出格式: table|json|yaml [默认: json]                                                                                                                                     │
│    --help                         显示此消息并退出                                                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**示例**:
```bash
dashscope rl test_functions "ro-ins--xxx" \
  --type rollout \
  --input "resources/rollout_input.json"
```

### 4.4 一站式工作流
自动注册函数、上传数据并提交作业。

```python
async def run(
    self,
    model: Optional[str] = None,
    training_datasets: Optional[List[TrainingDataset]] = None,
    validation_datasets: Optional[List[ValidationDataset]] = None,
    functions: Optional[
        Union[List[AgenticRLFunctionComponent], AgenticRLFunctionComponent]
    ] = None,
    hyper_parameters: Optional[Dict[str, str]] = None,
    resources: Optional[Dict[str, str]] = None,
    job_name: Optional[str] = None,
    teacher_model: Optional[str] = None,
    **kwargs,
) -> FineTune: ...
```
完整的工作流执行（注册 + 上传 + 提交）。

**参数**:
- `model`: 基础模型名称
- `training_datasets`: 训练数据集对象
- `validation_datasets`: 验证数据集对象
- `functions`: 函数组件
- `hyper_parameters`: 训练超参数
- `resources`: 训练资源配置
- `job_name`: 自定义作业名称
- `teacher_model`: Teacher 模型；设置后启用 OPD，Rollout 和 Reward 仍可省略

**返回**: `FineTune` 作业对象

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL, AgenticRLFunctionComponent, FunctionType, FunctionComponentModel

rl=AgenticRL()
rollout_runtime = {"cpu": 2, "memory_size": 4, "disk_size": 20960, "concurrency": 2,
                    "env": {}, "capacity": 5}
reward_runtimes = [
    {"cpu": 2, "memory_size": 4, "disk_size": 20960, "concurrency": 10, "env": {},
     "capacity": 8},
    {"cpu": 2, "memory_size": 4, "disk_size": 20960, "concurrency": 5, "env": {},
     "capacity": 6}
]
functions=[
    RolloutFunctionComponent(
        type=FunctionType.ROLLOUT,
        name="rollout-1",
        fcmodel=FunctionComponentModel(
            classpath="functions.rollout.rollout_only.DemoRolloutProcessor"),
        runtime=FunctionComponentRuntime(**rollout_runtime)),
    AgenticRLFunctionComponent(
        type=FunctionType.REWARD,
        name="reward-1",
        weight=1.0,
        fcmodel=FunctionComponentModel(
            classpath="functions.reward.reward.DemoRewardProcessor"),
        runtime=FunctionComponentRuntime(**reward_runtimes[0])),
]
training_datasets=[
    TrainingDataset(
        data_source_type=DataSourceType.FILE_ID,
        file_name="./data/calc_train_min.jsonl",
    ),
]
validation_datasets=[
    ValidationDataset(
        data_source_type=DataSourceType.FILE_ID,
        file_name="./data/calc_validation_min.jsonl",
    ),
]
job = await rl.run(
    model="qwen3.5-9b",
    training_datasets=training_datasets,
    validation_datasets=validation_datasets,
    functions=functions,
    hyper_parameters={'batch_size': '128'}
)
```

**[CLI] run**

**用法: dashscope rl run [OPTIONS]**
```bash
 🚀 启动完整的 RL 调优工作流（函数注册 → 数据集上传 → 作业提交）

 执行模式:
 1. 配置驱动: 使用 -c/--config 指定 YAML 文件
 2. 直接参数: 通过 CLI 选项提供所有必需参数

 必需参数:
 - training_files（至少一个）
 - 强化学习需要配置 Rollout 和 Reward，OPD 中两者均为可选。

╭─ 选项 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config                   -c      PATH   YAML 配置文件路径                                                                                                                                            │
│ --model                            TEXT   基础模型标识符                                                                                                                                                      │
│ --teacher-model                    TEXT   启用 OPD 并覆盖 YAML 中的 Teacher 模型；Rollout 和 Reward 可省略                                                                                                  │
│ --training-files                   TEXT   训练数据集文件路径                                                                                                                                            │
│ --validation-files                 TEXT   验证数据集文件路径                                                                                                                                          │
│ --rollout-classpath                TEXT   Rollout 类的 Python 导入路径 (module:Class)                                                                                                                         │
│ --reward-classpaths                TEXT   Reward 类路径列表 (file.py:ClassName)                                                                                                                             │
│ --group-reward-classpaths          TEXT   Group-reward 类路径列表 (file.py:ClassName)                                                                                                                       │
│ --rollout-name                     TEXT   预注册的 Rollout entity_name                                                                                                                                         │
│ --reward-names                     TEXT   Reward entity_names 的逗号分隔列表                                                                                                                                │
│ --group-reward-names               TEXT   Group-reward entity_names 的逗号分隔列表                                                                                                                          │
│ --rollout-weight                   TEXT   预注册的 Rollout entity_weight                                                                                                                                       │
│ --reward-weights                   FLOAT  Reward entity_weights 的逗号分隔列表                                                                                                                              │
│ --group-reward-weights             FLOAT  Group-reward entity_weights 的逗号分隔列表                                                                                                                        │
│ --reward-metric-weights            TEXT   Reward 指标权重的 JSON 字符串（字典列表）                                                                                                                       │
│ --rollout-runtime                  TEXT   Rollout 运行时配置的 JSON 字符串                                                                                                                                             │
│ --reward-runtimes                  TEXT   Reward 运行时配置的 JSON 字符串                                                                                                                                             │
│ --group-reward-runtimes            TEXT   Group-reward 运行时配置的 JSON 字符串                                                                                                                                       │
│ --hyper-parameters                 TEXT   超参数的 JSON 字符串                                                                                                                                             │
│ --job-name                         TEXT   调优作业的自定义名称                                                                                                                                             │
│ --api-key                          TEXT   DashScope API Key（如省略则使用 DASHSCOPE_API_KEY 环境变量） [环境变量: DASHSCOPE_API_KEY]                                                                                 │
│ --workspace-dir                    TEXT   作业工件的工作空间目录 [默认: ./]                                                                                                                        │
│ --output-format            -o      TEXT   输出格式: table|json|yaml [默认: table]                                                                                                                            │
│ --verbose                  -v             启用详细错误跟踪                                                                                                                                               │
│ --help                                    显示此消息并退出                                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**示例**: 运行完整工作流（自动）
```bash
dashscope rl run \
  --config "rl-job.yaml" \
  --verbose
```

### 4.5 作业管理

**[SDK] get**
```python
def get(cls, job_id: str, api_key: str = None, workspace: str = None, **kwargs) -> FineTune: ...
```
获取作业信息。

**参数**:
- `job_id`: 要检索的作业 ID
- `api_key`: 认证用的 API 密钥
- `workspace`: 工作空间标识符

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
job = AgenticRL.get("ft-12345")
```

**[CLI] get**

**用法: dashscope get [OPTIONS] JOB_ID**
```bash
 📊 查询特定作业的当前状态和元数据

╭─ 参数 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    job_id      TEXT  目标作业 ID [必需]                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 选项 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --api-key                TEXT  DashScope API Key（如省略则使用 DASHSCOPE_API_KEY 环境变量） [环境变量: DASHSCOPE_API_KEY]                                                                                            │
│ --output-format  -o      TEXT  [默认: table]                                                                                                                                                                      │
│ --help                         显示此消息并退出                                                                                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**示例**:
```bash
dashscope rl get "$JOB_ID" -o json
```

**[SDK] cancel**
```python
def cancel(cls, job_id: str, api_key: str = None, workspace: str = None, **kwargs) -> FineTuneCancel: ...
```
取消正在运行的作业。

**参数**:
- `job_id`: 要取消的作业 ID
- `api_key`: 认证用的 API 密钥
- `workspace`: 工作空间标识符

**示例**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
AgenticRL.cancel("ft-12345")
```

**[CLI] cancel**

**用法: dashscope cancel [OPTIONS] JOB_ID**
```bash
 🛑 取消正在运行的作业

╭─ 参数 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    job_id      TEXT  目标作业 ID [必需]                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 选项 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --api-key        TEXT  DashScope API Key（如省略则使用 DASHSCOPE_API_KEY 环境变量） [环境变量: DASHSCOPE_API_KEY]                                                                                                    │
│ --help                 显示此消息并退出                                                                                                                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**示例**:
```bash
dashscope rl cancel "$JOB_ID"
```

---

## 5. CLI 参考

CLI 镜像了 SDK 的功能。使用 `dashscope rl --help` 获取详细信息。

### 用法: dashscope [OPTIONS] COMMAND [ARGS]...
```bash

 🚀 Agentic RL 微调 CLI

╭─ 选项 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          显示此消息并退出                                                                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 命令 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ register_functions  🧩 注册 Rollout/Reward 函数组件，返回 entity_id & instance_id                                                                                                             │
│ test_functions      🧪 使用自定义输入数据测试已注册的 Rollout/Reward 函数实例                                                                                                               │
│ upload_data         📦 上传训练/验证数据集到平台，返回文件 ID                                                                                                                    │                                                                                                               │
│ run                 🚀 启动完整的 RL 调优工作流（函数注册 → 数据集上传 → 作业提交）                                                                                         │
│ status              📊 查询特定作业的当前状态和元数据                                                                                                                                  │
│ list                📋 分页列出历史微调作业                                                                                                                                         │
│ cancel              🛑 取消正在运行的作业                                                                                                                                     │
│ delete              🗑️ 删除作业记录（释放元数据）                                                                                                                                                  │
│ logs                📜 获取作业执行日志（支持分页）                                                                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## 6. 常见问题与故障排除

**Q: 函数注册失败**
*   **检查**：类路径是否正确 (`module.path:ClassName`)？
*   **检查**：工作空间根目录是否存在 `requirements.txt`？
*   **检查**：所有依赖项是否都列在 `requirements.txt` 中？

**Q: 作业提交失败**
*   **检查**：实体 ID 和文件 ID 是否有效？
*   **检查**：基础模型在您的区域是否可用？
*   **检查**：`reward_runtimes` 列表长度是否与 `reward_ids` 列表长度匹配？

**Q: 如何优化性能？**
*   对 I/O 密集型任务使用 `async def process`。
*   如果 CPU/内存允许，在运行时配置中增加 `concurrency`。
*   控制可观测 payload 的体积（避免过度输入/输出捕获）以降低开销。

**Q: 链路跟踪数据在哪里查看？**
*   在百炼控制台完成 ARMS 授权后，链路跟踪数据会导出到 **ARMS**。请确保 `requirements.txt` 已包含可观测依赖，并且使用了可观测接口（如 `observe_processor`、`trace_client`、`trace_tool` 等）。
*   **仍无数据时**：确认未将 `ENABLE_TRAJECTORY` 设为 `false`（该值会关闭链路跟踪）。
