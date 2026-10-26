# Agentic RL SDK/CLI 使用指南 [[English]](./README.md)

## 1. 安装 SDK

```bash
pip install dashscope>=1.25.19
```

## 2. 环境配置

### 2.1 设置环境变量

```bash
# 必填：API密钥（也可在代码中初始化: AgenticRL(api_key="for your api key") ）
export DASHSCOPE_API_KEY="your_api_key_here"

# 可选：日志级别设置info/debug/warning/critical（默认info）
export LOG_LEVEL="info"
```

### 2.2 配置依赖文件

> 注：`requirements.txt` 用于远端函数计算环境（Python >= 3.10）。本地调试时请确保使用 Python 3.10+。`dashscope` SDK 本身支持 Python 3.8+。

创建`requirements.txt`文件，包含以下核心依赖：

```requirements.txt
# 基础（必须）
dashscope>=1.25.19

# 框架依赖
fastapi==0.136.0
uvicorn==0.45.0
# 省略

# 轨迹函数依赖
langchain-core==1.3.0
langchain-mcp-adapters==0.2.2
langchain-openai==1.2.0
# 省略

# 添加其他自定义依赖...
```

## 3. 函数开发与数据准备

### 3.1 创建函数组件

在`functions`目录下开发函数：

- **奖励函数模板**：
    - `functions/reward/reward.py` - 基础实现
    - `functions/reward/reward_decorator.py` - 装饰器实现
- **轨迹函数模板**：
    - `functions/rollout/rollout.py` - 基础实现

> 注：functions/目录下需要包含__init__.py文件

函数组件是否必需取决于训练配置：

| 配置 | 场景 | 自定义 Rollout | 自定义 Reward |
|---|---|---:|---:|
| `rl-job.yaml` | 普通强化学习 | 必选 | 至少一个 |
| `opd-job.yaml` 不保留函数块 | 仅 Teacher | 否 | 否 |
| `opd-job.yaml` 仅保留 Reward | Teacher + Reward | 否 | 是 |
| `opd-job.yaml` 仅保留 Rollout | Teacher + Rollout | 是 | 否 |
| `opd-job.yaml` 保留两个函数块 | Teacher + Rollout + Reward | 是 | 是 |

### 3.2 准备训练数据

在`data`目录下添加数据集文件：

- `data/calc_training_min.jsonl` - 训练数据集（JSONL格式）
- `data/calc_validation_min.jsonl` - 验证数据集（JSONL格式）

## 4. 使用SDK执行任务

### 4.1 函数执行（注册+测试）

普通强化学习必须注册自定义 Rollout 和至少一个 Reward。OPD 的
Rollout、Reward 可选。

```bash
python test_functions.py
```

### 4.2 工作流执行（YAML配置+生命周期管理）

普通强化学习与 OPD 使用完全相同的 SDK 工作流，只需选择对应 YAML：

```python
from dashscope.finetune.agentic_rl import AgenticRL

client = AgenticRL()
# 普通强化学习使用 rl-job.yaml；OPD 使用 opd-job.yaml
client.init(config_path="rl-job.yaml")
result = await client.run()
```

`opd-job.yaml` 默认保留两个函数块，表示 Teacher + Rollout + Reward。删除
Reward 块表示 Teacher + Rollout；删除 Rollout 块表示 Teacher + Reward；
两个都删除表示仅 Teacher。

```bash
# submit_job.py 默认使用普通强化学习 rl-job.yaml
python submit_job.py
```

## 5. 使用CLI执行任务
示例代码：cli.sh

`cli.sh` 展示普通强化学习的完整流程。执行 OPD 时，对 `opd-job.yaml` 中已经
删除的函数块跳过对应注册和测试；数据提交、`rl run` 和任务生命周期命令
保持不变。

CLI 与 SDK 使用同一份 YAML：

```bash
# 普通强化学习
dashscope rl run -c rl-job.yaml

# OPD
dashscope rl run -c opd-job.yaml

# 覆盖 opd-job.yaml 中配置的 Teacher 模型
dashscope rl run -c opd-job.yaml \
  --teacher-model qwen3.5-397b-a17b
```

```bash
dashscope rl --help  # 查看完整命令帮助

 Usage: dashscope [OPTIONS] COMMAND [ARGS]...

 🚀 Agentic RL Fine-Tuning CLI

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ register_functions  🧩 Register Rollout/Reward function components, returns entity_id & instance_id                                                                                                             │
│ test_functions      🧪 Test a registered Rollout/Reward function instance with custom input data.                                                                                                               │
│ upload_data         📦 Upload training/validation datasets to the platform, returns file IDs                                                                                                                    │
│ run                 🚀 Launch the complete RL tuning workflow (function registration → dataset upload → job submission)                                                                                         │
│ get                 📊 Query the current status and metadata of a specific job                                                                                                                                  │
│ list                📋 List historical fine-tuning jobs with pagination                                                                                                                                         │
│ cancel              🛑 Cancel a running job                                                                                                                                                                     │
│ delete              🗑️ Delete a job record (releases metadata)                                                                                                                                                  │
│ logs                📜 Fetch job execution logs (supports pagination)                                                                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 最佳实践提示

1. **开发测试**：使用`test_functions`命令在提交前验证函数逻辑
2. **增量开发**：修改函数后重新注册即可，无需重建整个环境
3. **日志排查**：设置`LOG_LEVEL=debug`获取详细调试信息
4. **资源管理**：任务完成后使用`delete`命令释放资源

> 注：所有路径和参数需根据实际项目调整，示例脚本位于项目`workspace/`目录下
>
> 注：项目`workspace/`目录下的所有文件都会打包上传到远程进行在线计算，注意数据安全
>
> 注：项目`workspace/`目录下，设置上传排除的子目录和文件，参考环境变量：FC_ZIP_EXCLUDE_PATTERNS
>
> 注：项目`workspace/`目录下的所有文件打包上传限制大小：200M；可以通过环境变量FC_OSS_FILE_SIZE_WARNING修改
>
> 注：单个数据集文件（如训练/验证 JSONL）默认大小限制为 1G；可以通过环境变量 `DATASETS_FILE_SIZE_WARNING` 修改
>
> 注：如果要使用本地build的dashscope whl包（通过scripts/build.sh脚本生成），可以设置：
> export FC_PYPI_LIB="dashscope-1.25.19-py3-none-any.whl"，
> 并且放置在项目目录下workspace/；再把requirements.txt中dashscope依赖去掉。
