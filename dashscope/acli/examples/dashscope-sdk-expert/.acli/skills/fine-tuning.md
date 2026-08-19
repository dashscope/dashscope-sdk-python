---
name: fine-tuning
description: Quick reference for fine-tuning and deployment (FineTunes/Files/Deployments/AgenticRL) APIs, task workflows, inputs/outputs, and error codes
---

# Fine-tuning and Deployment

> Quick-reference note: all parameter names and return fields on this page follow the SDK source code (`dashscope/finetune/`, `dashscope/files.py`, `dashscope/common/`). Check the signatures on this page before answering; if unsure, verify on the spot with `inspect.signature` instead of relying on memory.

## Use Cases and Training Modes (SFT/CPT/DPO/AgenticRL)

| Mode | Entry Point | Notes |
| --- | --- | --- |
| SFT supervised fine-tuning | `FineTunes.call(..., mode="sft")` | `mode` is written into the request body as `training_type`; the source docstring lists values `sft` / `efficient_sft` |
| CPT continued pre-training / DPO preference optimization | `FineTunes.call` | The SDK does not enumerate CPT/DPO literals; pass them through `mode`/`hyper_parameters` per the server-side docs |
| AgenticRL reinforcement fine-tuning (GSPO) | `dashscope.finetune.agentic_rl.AgenticRL` | `training_type="reinforcement"`, `hyper_parameters["algorithm"]="gspo"` |

## Standard Workflow (upload file → create fine-tune job → poll status → deploy → invoke)

1. Upload file: `Files.upload(file_path, purpose="fine_tune")` → take `output["uploaded_files"][0]["file_id"]`
2. Create job: `FineTunes.call(model, training_file_ids=[file_id], hyper_parameters={...})` → take `output.job_id` (like `ft-xxx`)
3. Poll status: `FineTunes.get(job_id)` for polling, or `FineTunes.wait(job_id)` to block until a terminal state (polls every 30s); logs via `FineTunes.logs(job_id, offset=1, line=1000)`, event stream via `FineTunes.stream_events(job_id)`
4. Deploy: `Deployments.call(model=<finetuned_output>, capacity=N)` → `output.deployed_model`; use `Deployments.get(deployed_model)` and wait for `output.status` to become `RUNNING`
5. Invoke: the deployment name is the model name; use regular inference APIs such as `Generation.call(model=deployed_model, ...)`
6. Cleanup: `Deployments.scale(deployed_model, capacity)` to scale; `Deployments.delete(deployed_model)` to release; `FineTunes.cancel(job_id)` / `FineTunes.delete(job_id)` to cancel/delete a job

## SDK APIs (source of truth: the source code)

### FineTunes (`dashscope/finetune/finetunes.py`, SUB_PATH="fine-tunes")

The creation method is named `call` (the SDK has no `create` method):
`FineTunes.call(model, training_file_ids, validation_file_ids=None, mode=None, hyper_parameters={}, api_key=None, workspace=None, **kwargs) -> FineTune`

| Parameter | Type | Notes |
| --- | --- | --- |
| model | str | Base model to fine-tune |
| training_file_ids | list/str | Training file id; a str is auto-converted to a single-element list |
| validation_file_ids | list/str/None | Validation file id |
| mode | str/None | Training mode, written into the request body as `training_type` (`sft`/`efficient_sft`) |
| hyper_parameters | dict | Hyperparameters, e.g. `n_epochs`, `batch_size`, `learning_rate` |
| kwargs["finetuned_output"] | str | Custom output model name, passed through as `finetuned_output` in the request body |

Other methods: `get(job_id) -> FineTune`; `list(page_no=1, page_size=10) -> FineTuneList`; `cancel(job_id) -> FineTuneCancel`; `delete(job_id) -> FineTuneDelete`; `wait(job_id)` polls until `SUCCEEDED/FAILED/CANCELED`.

### Deployments (`dashscope/finetune/deployments.py`, SUB_PATH="deployments")

- `Deployments.call(model, capacity, version=None, suffix=None, ...) -> Deployment`: request body keys are `model_name/capacity/model_version/suffix`; fine-tuned models do not need `version`; with `suffix` set, the deployment name is `model_suffix`
- `get(deployed_model)`, `list(page_no=1, page_size=10)`, `delete(deployed_model)`, `scale(deployed_model, capacity)`

### Files (`dashscope/files.py`, SUB_PATH="files")

- `Files.upload(file_path, purpose="fine_tune", description=None, ...) -> DashScopeAPIResponse`: see `FilePurpose` for `purpose` values (`fine_tune`/`assistants`); for `fine_tune`, the JSONL is validated locally first and invalid files raise `InvalidFileFormat` directly
- `list(page=1, page_size=10)`, `get(file_id)`, `delete(file_id)`; all return `DashScopeAPIResponse`

### AgenticRL (`dashscope/finetune/agentic_rl.py`, brief)

- Usage: `client = AgenticRL(api_key=None)` → optionally `client.init(config_path="job.yaml")` (YAML-driven) → `await client.run(...)` completes "register functions → upload datasets → submit job" in one step; or call `register_functions()` / `upload_datasets()` / `submit_job()` step by step
- Key concepts: `RolloutFunctionComponent`/`RewardFunctionComponent` function components; `TrainingDataset`/`ValidationDataset` (`data_source_type`: `file_id`/`download_url`/`oss_mount`); `hyper_parameters` (`algorithm="gspo"`, `n_rollouts`, `kl_loss_coef`, `learning_rate`, etc.); `resources` (e.g. `mtu_spec_code`)
- Job management methods mirror FineTunes: `AgenticRL.get/list/cancel/delete/logs(job_id, ...)`

## Inputs/Outputs (JSONL format and job object fields)

- Training file: JSONL, one valid JSON object per line (the SDK validates line by line with `json.loads`); for SFT each line is usually `{"messages": [...]}`; RL data may carry a `rollout_extra` field
- `FineTune`: `status_code/request_id/code/message/output/usage`; `output` (`FineTuneOutput`) fields: `job_id, job_name, status, model, base_model, finetuned_output, training_file_ids, validation_file_ids, hyper_parameters, training_type, create_time, end_time, usage`
- `output.status` enum (`TaskStatus`): `PENDING/RUNNING/SUCCEEDED/FAILED/CANCELED/SUSPENDED/UNKNOWN`; after the job succeeds, use `output.finetuned_output` as the model name for deployment and invocation
- `FineTuneList.output`: `jobs` (List[FineTuneOutput]) + `page_no/page_size/total`
- `Deployment.output` (`DeploymentOutput`): `deployed_model, status, model_name, base_model, capacity, ready_capacity, charge_type, gmt_create, gmt_modified`; `status` (`DeploymentStatus`): `DEPLOYING/RUNNING(SERVING)/PENDING/DELETING/FAILED`

## Minimal Example (create an SFT job)

```python
import dashscope
from dashscope import Files, FineTunes

dashscope.api_key = "sk-xxx"  # or use the DASHSCOPE_API_KEY environment variable

up = Files.upload("train.jsonl", purpose="fine_tune")  # validates JSONL locally
file_id = up.output["uploaded_files"][0]["file_id"]

job = FineTunes.call(
    model="qwen-turbo",
    training_file_ids=[file_id],
    hyper_parameters={"n_epochs": 3},
)
if job.status_code == 200:
    rsp = FineTunes.wait(job.output.job_id)  # blocks until SUCCEEDED/FAILED/CANCELED
    print(rsp.output.status, rsp.output.finetuned_output)
else:
    print(job.code, job.message)  # on failure read code/message/request_id
```

## Common Error Codes

| Error code/exception | HTTP status | Meaning | Handling |
| --- | --- | --- | --- |
| `InvalidFileFormat` | none (local exception) | Uploaded file is not valid JSONL | Fix the JSON line by line and re-upload |
| `AuthenticationError` | 401 | API Key missing/invalid | Set `DASHSCOPE_API_KEY` or pass `api_key` explicitly |
| Response `code`/`message` | 403 | No access to the file/job/deployment | Check the `workspace` and resource ownership |
| Response `code`/`message` | 404 | `job_id`/`file_id`/`deployed_model` does not exist | Verify the id with the corresponding `list()` |
| `REPEATABLE_STATUS` | 503/504 | Service temporarily unavailable | Retry, or keep polling with `FineTunes.wait` |
| AgenticRL `error_code` 3001-3008 | - | 3001 key config, 3002 function registration, 3003 dataset upload, 3004 duplicate function name, 3005 job submission, 3006 flow failure, 3007 unsupported function type, 3008 function validation failure | Use the exception's `root_cause` to locate the root cause |

General error-handling pattern: first check `status_code == 200` on the response object; on failure read `code`/`message`/`request_id`; the base class for SDK local exceptions is `dashscope.common.error.DashScopeException`.

## Java SDK

The Java SDK index (v2.22.23) has no FineTunes/Files/Deployments wrappers — for Java, call the Bailian HTTP API directly (endpoints identical to the ones behind the Python calls above).
