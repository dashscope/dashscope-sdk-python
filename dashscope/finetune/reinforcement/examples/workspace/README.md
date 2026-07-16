# Agentic RL SDK/CLI User Guide [[中文]](./README-zh.md)

## 1. Install SDK

```bash
pip install dashscope>=1.25.19
```

## 2. Environment Configuration

### 2.1 Setting Environment Variables

```bash
# Required: API key (can also be initialized in code: AgenticRL(api_key="for your api key") )
export DASHSCOPE_API_KEY="your_api_key_here"

# Optional: Log level setting info/debug/warning/critical (default info)
export LOG_LEVEL="info"
```

### 2.2 Configuring Dependencies File

> Note: The `requirements.txt` is used for the remote Function Compute environment (Python >= 3.10). For local debugging, please ensure you use Python 3.10+. The `dashscope` SDK itself supports Python 3.8+.

Create a `requirements.txt` file with the following core dependencies:

```requirements.txt
# Base (Must)
dashscope>=1.25.19

# Framework dependencies
fastapi==0.136.0
uvicorn==0.45.0
# ...

# Trajectory function dependencies
langchain-core==1.3.0
langchain-mcp-adapters==0.2.2
langchain-openai==1.2.0
# ...

# Add other custom dependencies...
```

## 3. Function Development and Data Preparation

### 3.1 Creating Function Components

Develop functions under the `functions` directory:

- **Reward function templates**:
    - `functions/reward/reward.py` - Basic implementation
    - `functions/reward/reward_decorator.py` - Decorator implementation
- **Rollout function templates**:
    - `functions/rollout/rollout.py` - Basic implementation

> Note: The `functions/` directory must contain an `__init__.py` file

Function components depend on the selected training configuration:

| Configuration | Mode | Custom Rollout | Custom Reward |
|---|---|---:|---:|
| `rl-job.yaml` | Regular reinforcement | Required | At least one |
| `opd-job.yaml` without function blocks | Teacher only | No | No |
| `opd-job.yaml` with Reward only | Teacher + Reward | No | Yes |
| `opd-job.yaml` with Rollout only | Teacher + Rollout | Yes | No |
| `opd-job.yaml` with both blocks | Teacher + Rollout + Reward | Yes | Yes |

### 3.2 Preparing Training Data

Add dataset files under the `data` directory:

- `data/calc_training_min.jsonl` - Training dataset (JSONL format)
- `data/calc_validation_min.jsonl` - Validation dataset (JSONL format)

## 4. Executing Tasks with SDK

### 4.1 Function Execution (Register + Test)

Regular reinforcement training requires a custom Rollout and at least one
Reward. Rollout and Reward are optional for OPD.

```bash
python test_functions.py
```

### 4.2 Workflow Execution (YAML Config + Lifecycle Management)

Regular reinforcement training and OPD use the same SDK workflow. Select the
corresponding YAML:

```python
from dashscope.finetune.agentic_rl import AgenticRL

client = AgenticRL()
# Use rl-job.yaml for regular reinforcement or opd-job.yaml for OPD
client.init(config_path="rl-job.yaml")
result = await client.run()
```

`opd-job.yaml` keeps both function blocks by default, representing Teacher +
Rollout + Reward. Remove the Reward block for Teacher + Rollout, remove the
Rollout block for Teacher + Reward, or remove both for Teacher only.

```bash
# submit_job.py uses the regular reinforcement rl-job.yaml by default
python submit_job.py
```

## 5. Executing Tasks with CLI
Example code：cli.sh

`cli.sh` demonstrates the regular reinforcement workflow. For OPD, skip
registration and testing for function blocks removed from `opd-job.yaml`; dataset
submission, `rl run`, and job lifecycle commands remain unchanged.

The CLI consumes the same YAML files as the SDK:

```bash
# Regular reinforcement
dashscope rl run -c rl-job.yaml

# OPD
dashscope rl run -c opd-job.yaml

# Override the Teacher configured in opd-job.yaml
dashscope rl run -c opd-job.yaml \
  --teacher-model qwen3.5-397b-a17b
```

```bash
dashscope rl --help  # View full command help

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

## Best Practice Tips

1. **Development & Testing**: Use the `test_functions` command to verify function logic before submission
2. **Incremental Development**: Simply re-register functions after modification, no need to rebuild the entire
   environment
3. **Log Troubleshooting**: Set `LOG_LEVEL=debug` to obtain detailed debug information
4. **Resource Management**: Use the `delete` command to release resources after task completion

> Note: All paths and parameters should be adjusted according to the actual project; example scripts are located in the
> project's `workspace/` directory
>
> Note: All files under the project's `workspace/` directory will be packaged and uploaded to the cloud for online
> computation; be mindful of data security
>
> Note: To set excluded subdirectories and files under the project's `workspace/` directory for upload, refer to the
> environment variable: `FC_ZIP_EXCLUDE_PATTERNS`
>
> Note: The total size limit for packaging and uploading all files under the project's `workspace/` directory is 200M;
> this can be modified via the environment variable `FC_OSS_FILE_SIZE_WARNING`
>
> Note: The maximum size for a single dataset file (e.g., training/validation JSONL) is 1G by default;
> this can be modified via the environment variable `DATASETS_FILE_SIZE_WARNING`
>
> Note: If you want to use a locally built dashscope whl package (generated via the scripts/build.sh script), you can set:
> export FC_PYPI_LIB="dashscope-1.25.19-py3-none-any.whl"
> and place it in the workspace/workspace/directory under the project root. Also remove the dashscope dependency fromrequirements.txt`.
