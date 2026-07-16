# Agentic Reinforcement Learning: User Guide [[中文]](./UserGuide-zh.md)

---

## 1. Introduction

The **Agentic RL SDK/CLI** provides a comprehensive toolchain for building, training, and managing Reinforcement Learning (RL) models for Large Language Models (LLMs). It simplifies the complex workflow of defining agent behaviors, collecting trajectories, and optimizing policies.

The SDK consists of two core modules:
1.  **Functions Module**: Manages custom Python code for **Rollout** (trajectory generation), **Reward** (scoring), and **Group Reward** (batch scoring). It supports automatic registration, testing, and built-in **Observability (Tracing)**.
2.  **Tuning Module**: Handles dataset management, hyperparameter configuration, job submission, and lifecycle management (status, logs, cancellation).

**Workflow**:
1. Register custom functions. Regular reinforcement training requires a
   Rollout and at least one Reward; Rollout and Reward are optional for OPD.
2. Use the Tuning Module to upload datasets, configure the job, and submit it
   for training.

---

## 2. Installation & Setup

### 2.1 Install via PyPI
```bash
pip install dashscope>=1.25.19
```

### 2.2 Install from Source (For Development)
```bash
git clone https://github.com/dashscope/dashscope-sdk-python.git
cd dashscope-sdk-python
pip install -e .  # Install in editable mode for development
```

### 2.3 Authorization
1.  Obtain your **DashScope API Key**.
2.  Set the environment variable:
    ```bash
    export DASHSCOPE_API_KEY='your-api-key-here'
    ```

### 2.4 Project Structure
A recommended workspace structure ensures smooth deployment and local testing:

```text
workspace/
├── data/                   # Datasets
│   ├── training.jsonl
│   └── validation.jsonl
├── functions/              # Custom functions; optional only for OPD
│   ├── reward/
│   │   ├── group_reward.py
│   │   └── reward.py
│   └── rollout/
│       └── rollout.py
├── requirements.txt        # Dependencies for Function Components
├── rl-job.yaml             # Regular Agentic RL configuration
└── opd-job.yaml            # OPD configuration
```

Regular reinforcement training requires a custom Rollout and at least one
Reward. Only OPD may omit Rollout and Reward, or configure either one
independently.

### 2.5 Dependency Packages (requirements.txt Guidelines)
This file is **mandatory** for deploying Function Components to the cloud. It must reside in the workspace root.

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
```

**Key Notes:**
*   **Default Packages**: `dashscope` are pre-installed in the runtime environment. Do NOT include them in your `requirements.txt`.
*   **Protobuf**: Must be within the specified range to avoid compatibility issues.

#### Dependencies for Observability (Tracing)
To use observability spans (processor / LLM / tool), add the following dependencies to your `requirements.txt` (recommended pinned versions for reproducible deployments):

```txt
opentelemetry-api==1.41.1
opentelemetry-sdk==1.41.1
opentelemetry-exporter-otlp-proto-http==1.41.1
opentelemetry-processor-baggage==0.62b1
loongsuite-util-genai==0.4.0
```

### 2.6 Logging Configuration
Configure logging verbosity via the `LOG_LEVEL` environment variable:

```bash
export LOG_LEVEL="DEBUG"   # Most verbose (masks sensitive info like API keys)
export LOG_LEVEL="INFO"    # Default
export LOG_LEVEL="WARNING"
export LOG_LEVEL="ERROR"
export LOG_LEVEL="CRITICAL"
```

---

## 3. Writing Functions
Reference: workspace/functions directory, output rollout.py, reward.py, etc.

### 3.1 Implementing Functions

Functions must inherit from the abstract base classes provided by the SDK.

#### Rollout Processor
Generates agent trajectories (interactions with the environment/LLM).

```python
from dashscope.finetune.reinforcement import AbstractRolloutProcessor, RolloutInput, RolloutOutput

class DemoRolloutProcessor(AbstractRolloutProcessor):
    async def process(self, input: RolloutInput) -> RolloutOutput:
        # Generate trajectory (supports async/def)
        pass
```

#### Reward Processor
Scores individual steps or final outputs.

```python
from dashscope.finetune.reinforcement import AbstractRewardProcessor, RewardInput, RewardOutput

class DemoRewardProcessor(AbstractRewardProcessor):
    def process(self, input: RewardInput) -> RewardOutput:
        # Calculate reward score
        pass
```

#### Advanced Reward Processor with Decorators
```python
from dashscope.finetune.reinforcement import reward_func, sub_reward_func, aggregate_func

@reward_func("SafetyProcessor")
class SafetyProcessor(AbstractRewardProcessor):
    @sub_reward_func("toxicity", sub_weight=0.7)
    def toxicity(self, input: RewardInput) -> RewardOutput: ...

    @sub_reward_func("refusal", sub_weight=0.3)
    async def refusal(self, input: RewardInput) -> RewardOutput: ...

    @aggregate_func
    async def aggregate(self, sub_rewards: dict[str, RewardOutput]) -> RewardOutput: # Custom aggregation logic
        weights = self.get_weights()
        scores = self.get_scores(sub_rewards)
        reward_metrics = self.get_reward_metrics(sub_rewards)

        total = ...  # Calculate total reward
        return RewardOutput(...)
```

#### Group Reward Processor
Scores a batch of trajectories collectively (e.g., for ranking).

```python
from dashscope.finetune.reinforcement import AbstractGroupRewardProcessor, GroupRewardInput, GroupRewardOutput

class DemoGroupRewardProcessor(AbstractGroupRewardProcessor):
    def setup(self) -> None:
        pass

    async def process(self, input: GroupRewardInput) -> GroupRewardOutput:
        # Calculate group rewards
        pass
```

### 3.2 Observability (Tracing)

Enable deep visibility into your agent's execution using OpenTelemetry. Trace data is exported to **ARMS** (Alibaba Cloud Real-Time Monitoring Service) after ARMS authorization is completed in the console. See [ARMS documentation](https://help.aliyun.com/zh/arms/?spm=5176.30275541.J_ZGek9Blx07Hclc3Ddt9dg.3.3ce02f3dmKOpPK&scm=20140722.S_card@@%E4%BA%A7%E5%93%81@@596792.S_new~UND~card.ID_card@@%E4%BA%A7%E5%93%81@@596792-RL_arms-LOC_2024SPSearchCard-OR_ser-PAR1_0bc1409817757870159831522e3953-V_4-RE_new5-P0_0-P1_0).

#### Prerequisites
1.  Add observability dependencies to `requirements.txt` (see Section 2.5, “Dependencies for Observability (Tracing)”).

#### Instrumentation Decorators
Import from `dashscope.finetune.reinforcement.component.observability`.

| Decorator | Usage | Description |
| :--- | :--- | :--- |
| `@observe_processor` | On `process()` method | Auto-traces input/output, latency, and status. Determines span kind (ROLLOUT/REWARD) automatically. |
| `trace_client()` | In `setup()` or before first LLM call | Wraps LLM clients (OpenAI, DashScope, LangChain-like). Auto-traces all downstream calls. **Recommended.** |
| `@observe_llm` | On custom LLM funcs | Use if `trace_client` doesn't support your wrapper. Requires `model` and `messages` as kwargs. |
| `trace_tool()` | In `setup()` (after tools are created) | Wraps tools (LangChain/MCP/LangGraph-like). Auto-traces tool invocations. |
| `@observe_tool` | On plain functions | Use for simple Python functions not wrapped as BaseTools. |

> **Setup:** Called once on server startup; supports both sync and async. Sync setup is offloaded to avoid blocking the event loop.

#### What `trace_client()` supports (duck typing)
`trace_client(client)` is detected by structure (not by class name). It supports:

- **Full OpenAI client** exposing `.chat.completions.create`
- **Completions resource** exposing `.create` and **no** `.chat` (e.g. `ChatOpenAI.client`)
- **LangChain-like wrapper** exposing `.client` and/or `.async_client`
- **DashScope Generation** class (pass the class itself, which has `call` as a classmethod)

#### What `trace_tool()` supports
`trace_tool(tools)` accepts the following shapes:

- A single tool object (e.g. LangChain `BaseTool`)
- A list/tuple of tools
- A dict mapping names to tools
- A LangGraph `ToolNode` (tools are expanded internally)
- MCP tools returned by `langchain-mcp-adapters` (provider is auto-set to `"mcp"`)

> **MCP note:** MCP Server and Client run in separate processes. `@observe_tool` on the Server-side function has no effect on the Client side. Always call `trace_tool(tools)` after `get_tools()` on the Client side.

#### Example: Instrumented Rollout Processor

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
        # 1. Trace LLM Client
        self._client = openai.AsyncOpenAI(base_url="...", api_key="...")
        trace_client(self._client)

        # 2. Trace Tools (e.g., MCP)
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({...})
        self._tools = await client.get_tools()
        trace_tool(self._tools)

    @observe_processor
    async def process(self, input: RolloutInput) -> RolloutOutput:
        messages = input.messages or []
        model = input.model_resource.model_name

        # This call is automatically traced due to trace_client(self._client)
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

---

## 4. SDK & CLI Reference
* [SDK] Interface class: dashscope.finetune.agentic_rl.AgenticRL
* [CLI] Entry point: dashscope rl

### 4.1 Job Configuration

Regular reinforcement training and OPD use the same SDK/CLI workflow. Select
the corresponding YAML configuration; code arguments override YAML values.

| Training type | Configuration | Custom functions |
|---|---|---|
| Regular reinforcement | `rl-job.yaml` | Rollout is required, with at least one Reward |
| OPD | `opd-job.yaml` | Rollout and Reward are independently optional and may both be omitted |

**[SDK] \_\_init\_\_**
```python
def __init__(self, api_key: str = None): ...
```
Initializes the AgenticRL instance.

**Parameters**:
- `api_key`: API key for authentication (uses environment variable if not provided)

**Example**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
rl = AgenticRL(api_key="your_api_key")
```

**[SDK] [init](submit_job.py)**
```python
def init(self, config_path: Optional[str] = None, **kwargs) -> Self: ...
```
Initializes the instance from a YAML configuration file.

**Parameters**:
- `config_path`: Path to YAML configuration file
- `**kwargs`: Configuration overrides

**Returns**: Self instance

**Example**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
config_path = "opd-job.yaml"  # Use "rl-job.yaml" for regular reinforcement
rl = AgenticRL().init(config_path, job_name="custom_job")
result = await rl.run()
```

When `opd-job.yaml` is selected, keep or remove the Rollout and Reward blocks
to choose the required capabilities:

| Function blocks kept | OPD mode | Trajectory source | Task reward |
|---|---|---|---|
| Neither | Teacher only | Platform | Disabled |
| Reward only | Teacher + Reward | Platform | Custom Reward |
| Rollout only | Teacher + Rollout | Custom Rollout | Disabled |
| Rollout and Reward | Teacher + Rollout + Reward | Custom Rollout | Custom Reward |

The committed `opd-job.yaml` keeps both blocks. Delete either block, or both,
to select another OPD combination:

```yaml
teacher_model: qwen3.5-397b-a17b

functions:
# Remove this block to use platform generation.
- type: rollout
  # ...
# Remove this block to disable custom task rewards.
- type: reward
  # ...

training:
  type: pg-opd
```

```bash
# Regular reinforcement
dashscope rl run -c rl-job.yaml

# OPD
dashscope rl run -c opd-job.yaml
```

### 4.2 Registering Functions

Uploads code and registers Function Components (functions).

**[SDK] [register_functions](submit_job.py)**
```python
def register_functions(self, functions: Optional[Union[List[Union[RolloutFunctionComponent, RewardFunctionComponent]], RolloutFunctionComponent, RewardFunctionComponent]] = None, lazy_load: Optional[bool] = True) -> tuple: ...
```
Registers function components.

**Parameters**:
- `functions`: Function components to register
- `lazy_load`: Defer loading until execution

**Returns**: Tuple of entity/instance IDs

**Example**:
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
                classpath="functions.rollout.rollout2.DemoRolloutProcessor"),
        ),

        AgenticRLFunctionComponent(
            type=FunctionType.REWARD,
            fcmodel=FunctionComponentModel(
                classpath="functions/reward/reward.py:DemoRewardProcessor"),
        ),

        AgenticRLFunctionComponent(
            type=FunctionType.GROUP_REWARD,
            fcmodel=FunctionComponentModel(
                classpath="functions.reward.group_reward.DemoGroupRewardProcessor"),
        ),
    ],
    lazy_load=False  # Set False to get instance IDs immediately for testing
)
```

**[CLI] register_functions**

**Usage: dashscope register_functions [OPTIONS]**
```bash
 🧩 Register Rollout/Reward function components, returns entity_id & instance_id

 Requires at least one of:
 - rollout_classpath
 - reward_classpaths

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --rollout-classpaths                             TEXT  List for rollout class path (file.py:ClassName)                                                                                                               │
│ --reward-classpaths                              TEXT  List for reward class path (file.py:ClassName)                                                                                                                │
│ --group-reward-classpaths                        TEXT  List for group-reward class path (file.py:ClassName)                                                                                                          │
│ --workspace-dir                                  TEXT  Local workspace directory [default: ./]                                                                                                                       │
│ --lazy-load                    --no-lazy-load          Delay instance loading (set False for debugging) [default: lazy-load]                                                                                         │
│ --api-key                                        TEXT  DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted) [env var: DASHSCOPE_API_KEY]                                                                    │
│ --output-format            -o                    TEXT  Output format: table|json|yaml [default: json]                                                                                                                │
│ --help                                                 Show this message and exit.                                                                                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
**Example**:
```bash
dashscope rl register_functions \
  --rollout-classpath "functions.rollout.rollout2.DemoRolloutProcessor" \
  --group-reward-classpaths "functions.reward.group_reward.DemoGroupRewardProcessor" \
  --workspace-dir "./" \
  --output-format json
```

### 4.3 Testing Functions

#### 4.3.1 Remote Testing
**[SDK] [test_functions](test_functions.py)**

Test registered instances with sample data.

```python
def test_functions(cls, instance_id: str, type: FunctionType, input_data: Dict[str, Any], api_key: str = None): ...
```

**Parameters**:
- `instance_id`: Function instance ID
- `type`: Function type (ROLLOUT/REWARD/GROUP_REWARD)
- `input_data`: Test input data
- `api_key`: API key for authentication

**Example**:
```python
from dashscope.finetune.agentic_rl import AgenticRL, FunctionType

# Test Rollout
result = await AgenticRL.test_functions(
    instance_id=rollout_iids[0],
    functype=FunctionType.ROLLOUT,
    input_data="resouces/rollout_input.json" # path to JSON file
)

# Test Reward
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

**Usage: dashscope test_functions [OPTIONS] INSTANCE_ID**
```bash
 🧪 Test a registered Rollout/Reward function instance with custom input data.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    instance_id      TEXT  Target function instance ID (e.g., ro-ins-xxx or rw-ins-xxx) [required]                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --type           -t      TEXT  Function type: ROLLOUT or REWARD [required]                                                                                                                                        │
│ *  --input          -i      TEXT  JSON string or file path containing test payload [required]                                                                                                                        │
│    --api-key                TEXT  DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted) [env var: DASHSCOPE_API_KEY]                                                                                         │
│    --output-format  -o      TEXT  Output format: table|json|yaml [default: json]                                                                                                                                     │
│    --help                         Show this message and exit.                                                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Example**:
```bash
dashscope rl test_functions "ro-ins-xxx" \
  --type rollout \
  --input "resources/rollout_input.json"
```

### 4.4 One-Step Workflow
Automatically registers functions, uploads data, and submits the job.

**[SDK] [run](submit_job.py)**
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
Full workflow execution (registration + upload + submission).

**Parameters**:
- `model`: Base model name
- `training_datasets`: Training dataset objects
- `validation_datasets`: Validation dataset objects
- `functions`: Function components
- `hyper_parameters`: Training hyper_parameters
- `resources`: Training resource configuration
- `job_name`: Custom job name
- `teacher_model`: Teacher model; enables OPD while Rollout and Reward remain optional

**Returns**: `FineTune` job object

**Example**:
```python
from dashscope.finetune.agentic_rl import AgenticRL, AgenticRLFunctionComponent, FunctionType, FunctionComponentModel

rl=AgenticRL()
rollout_runtime = {"cpu": 2, "memory_size": 4096, "disk_size": 20960, "concurrency": 2,
                    "env": {}, "capacity": 5}
reward_runtimes = [
    {"cpu": 2, "memory_size": 4096, "disk_size": 20960, "concurrency": 10, "env": {},
     "capacity": 8},
    {"cpu": 2, "memory_size": 4096, "disk_size": 20960, "concurrency": 5, "env": {},
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

**Usage: dashscope rl run [OPTIONS]**
```bash
 🚀 Launch the complete RL tuning workflow (function registration → dataset upload → job submission)

 Execution modes:
 1. Configuration-driven: Use -c/--config to specify a YAML file
 2. Direct parameter: Provide all required arguments via CLI options

 Required parameters:
 - training_files (at least one)
 - Rollout and Reward are required for reinforcement learning, but optional for OPD.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config                   -c      PATH   Path to YAML configuration file                                                                                                                                            │
│ --model                            TEXT   Base model identifier                                                                                                                                                      │
│ --teacher-model                    TEXT   Enable OPD and override the Teacher model in YAML; Rollout and Reward are optional                                                                                          │
│ --training-files                   TEXT   Paths to training dataset files                                                                                                                                            │
│ --validation-files                 TEXT   Paths to validation dataset files                                                                                                                                          │
│ --rollout-classpath                TEXT   Python import path to rollout class (module:Class)                                                                                                                         │
│ --reward-classpaths                TEXT   List for reward class path (file.py:ClassName)                                                                                                                             │
│ --group-reward-classpaths          TEXT   List for group-reward class path (file.py:ClassName)                                                                                                                       │
│ --rollout-name                     TEXT   Pre-registered Rollout entity_name                                                                                                                                         │
│ --reward-names                     TEXT   Comma-separated list of reward entity_names                                                                                                                                │
│ --group-reward-names               TEXT   Comma-separated list of group-reward entity_names                                                                                                                          │
│ --rollout-weight                   TEXT   Pre-registered Rollout entity_weight                                                                                                                                       │
│ --reward-weights                   FLOAT  Comma-separated list of reward entity_weights                                                                                                                              │
│ --group-reward-weights             FLOAT  Comma-separated list of group-reward entity_weights                                                                                                                        │
│ --reward-metric-weights            TEXT   Reward metric weights as JSON string (list of dicts)                                                                                                                       │
│ --rollout-runtime                  TEXT   Rollout runtime as JSON string                                                                                                                                             │
│ --reward-runtimes                  TEXT   Reward runtimes as JSON string                                                                                                                                             │
│ --group-reward-runtimes            TEXT   Group-reward runtimes as JSON string                                                                                                                                       │
│ --hyper-parameters                 TEXT   JSON string of hyper_parameters                                                                                                                                             │
│ --job-name                         TEXT   Custom name for the tuning job                                                                                                                                             │
│ --api-key                          TEXT   DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted) [env var: DASHSCOPE_API_KEY]                                                                                 │
│ --workspace-dir                    TEXT   Workspace directory for job artifacts [default: ./]                                                                                                                        │
│ --output-format            -o      TEXT   Output format: table|json|yaml [default: table]                                                                                                                            │
│ --verbose                  -v             Enable detailed error traces                                                                                                                                               │
│ --help                                    Show this message and exit.                                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Example**: Run Full Workflow (Auto)
```bash
dashscope rl run \
  --config "rl-job.yaml" \
  --verbose
```

### 4.5 Job Management

**[SDK] get**
```python
def get(cls, job_id: str, api_key: str = None, workspace: str = None, **kwargs) -> FineTune: ...
```
Gets job information.

**Parameters**:
- `job_id`: ID of job to retrieve
- `api_key`: API key for authentication
- `workspace`: Workspace identifier

**Example**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
job = AgenticRL.get("job-12345")
```

**[CLI] get**

**Usage: dashscope get [OPTIONS] JOB_ID**
```bash
 📊 Query the current status and metadata of a specific job

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    job_id      TEXT  Target job ID [required]                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --api-key                TEXT  DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted) [env var: DASHSCOPE_API_KEY]                                                                                            │
│ --output-format  -o      TEXT  [default: table]                                                                                                                                                                      │
│ --help                         Show this message and exit.                                                                                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Example**:
```bash
dashscope rl get "$JOB_ID" -o json
```

**[SDK] cancel**
```python
def cancel(cls, job_id: str, api_key: str = None, workspace: str = None, **kwargs) -> FineTuneCancel: ...
```
Cancels a running job.

**Parameters**:
- `job_id`: ID of job to cancel
- `api_key`: API key for authentication
- `workspace`: Workspace identifier

**Example**:
```python
from dashscope.finetune.agentic_rl import AgenticRL
AgenticRL.cancel("job-12345")
```

**[CLI] cancel**

**Usage: dashscope cancel [OPTIONS] JOB_ID**
```bash
 🛑 Cancel a running job

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    job_id      TEXT  Target job ID [required]                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --api-key        TEXT  DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted) [env var: DASHSCOPE_API_KEY]                                                                                                    │
│ --help                 Show this message and exit.                                                                                                                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Example**:
```bash
dashscope rl cancel "$JOB_ID"
```

---

## 5. CLI Reference

The CLI mirrors the SDK functionality. Use `dashscope rl --help` for details.

### Usage: dashscope [OPTIONS] COMMAND [ARGS]...
```bash

 🚀 Agentic RL Fine-Tuning CLI

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ register_functions  🧩 Register Rollout/Reward function components, returns entity_id & instance_id                                                                                                             │
│ test_functions      🧪 Test a registered Rollout/Reward function instance with custom input data.                                                                                                               │
│ upload_data         📦 Upload training/validation datasets to the platform, returns file IDs                                                                                                                    │
│ submit              📤 Submit fine-tuning job (requires pre-registered functions & uploaded datasets)                                                                                                                 │
│ run                 🚀 Launch the complete RL tuning workflow (function registration → dataset upload → job submission)                                                                                         │
│ status              📊 Query the current status and metadata of a specific job                                                                                                                                  │
│ list                📋 List historical fine-tuning jobs with pagination                                                                                                                                         │
│ cancel              🛑 Cancel a running job                                                                                                                                                                     │
│ delete              🗑️ Delete a job record (releases metadata)                                                                                                                                                  │
│ logs                📜 Fetch job execution logs (supports pagination)                                                                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## 6. FAQ & Troubleshooting

**Q: Function registration fails.**
*   **Check**: Is the classpath correct (`module.path:ClassName`)?
*   **Check**: Does `requirements.txt` exist in the workspace root?
*   **Check**: Are all dependencies listed in `requirements.txt`?

**Q: Job submission fails.**
*   **Check**: Are the Entity IDs and File IDs valid?
*   **Check**: Is the base model available in your region?
*   **Check**: Do `reward_runtimes` list length match `reward_ids` list length?

**Q: How to optimize performance?**
*   Use `async def process` for I/O bound tasks.
*   Increase `concurrency` in runtime config if CPU/Memory allows.
*   Keep observability payloads bounded (avoid excessive input/output capture) to reduce overhead.

**Q: Where are my traces?**
*   Traces are sent to **ARMS** after ARMS authorization is completed in the Bailian Console. Ensure your `requirements.txt` includes the observability dependencies and that you are using the observability APIs (`observe_processor`, `trace_client`, `trace_tool`, etc.).
