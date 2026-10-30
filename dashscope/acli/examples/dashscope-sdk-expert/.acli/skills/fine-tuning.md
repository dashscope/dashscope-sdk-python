---
name: fine-tuning
description: Quick reference for fine-tuning and deployment (FineTunes/Files/Deployments/AgenticRL) APIs, task workflows, inputs/outputs, and error codes
---

# Fine-tuning and Deployment

> Python SDK 2.0.3 uses `dashscope.Client` / `dashscope.AsyncClient` at the package root (`dashscope.v2` does not exist). Verify current signatures in `dashscope/resources/`, workflow code in `dashscope/applications/finetune/`, and old implementations in `dashscope/legacy_1x/finetune/`; use `inspect.signature` when unsure. New examples below use flat response fields and close their clients.

## Use Cases and Training Modes

| Mode | Current entry point | Notes |
| --- | --- | --- |
| SFT supervised fine-tuning | `client.fine_tunes.create(..., mode="sft")` | `mode` is sent as `training_type`; source documents `sft` / `efficient_sft` |
| CPT / DPO | `client.fine_tunes.create(...)` | The SDK does not enumerate all server literals; pass supported `mode` and `hyper_parameters` values through |
| Agentic RL (including GSPO configuration) | `client.rl` and `client.rl.jobs` | Current client workflow and job-management facade |
| Direct AgenticRL workflow | `dashscope.finetune.agentic_rl.AgenticRL` | Separate workflow object; do not treat it as `client.rl` or as a FineTunes manager |

## Standard SFT Workflow

1. Upload JSONL with `client.files.upload(file_path=..., purpose="fine-tune")` and read `File.file_id`.
2. Create a job with `client.fine_tunes.create(...)`; submission returns a `FineTune` snapshot with `job_id` and `status`.
3. Poll with `get(job_id)` or block with `wait(job_or_job_id)`. Read logs with `logs()` or consume SSE with `stream_events()`.
4. After success, deploy `FineTune.finetuned_model` with `client.deployments.create(model=..., capacity=...)`; poll `get()` until the desired deployment status.
5. Invoke the deployment by using `Deployment.deployed_model` as the model name in the appropriate inference API.
6. Scale or release with `client.deployments.scale/delete`; cancel or delete training with `client.fine_tunes.cancel/delete`.

```python
import dashscope

with dashscope.Client() as client:  # reads DASHSCOPE_API_KEY by default
    uploaded = client.files.upload(file_path="train.jsonl")

    submitted = client.fine_tunes.create(
        model="qwen3.8-32b",
        training_file_ids=[uploaded.file_id],
        hyper_parameters={"n_epochs": 3},
    )
    job = client.fine_tunes.wait(submitted)
    print(job.job_id, job.status, job.finetuned_model)

    deployment = client.deployments.create(
        model=job.finetuned_model,
        capacity=1,
        suffix="demo",
    )
    print(deployment.deployed_model, deployment.status)
```

`AsyncClient` provides awaitable file, fine-tune, deployment, and RL resource methods; use `async with dashscope.AsyncClient()` and close streams with their async cleanup API.

## Files (`client.files`)

| Method | Return / important fields |
| --- | --- |
| `upload(*, file_path, purpose="fine-tune", description=None, ...)` | `File`: `file_id`, `name`, `purpose`, `size`, `uploaded_at` |
| `list(*, page_no=1, page_size=10, ...)` | `FileList`: `files`, `page_no`, `page_size`, `total_count` |
| `get(file_id, ...)` | `File` |
| `delete(file_id, ...)` | `DeletionStatus`, including `deleted` |

Fine-tune uploads validate JSONL locally, one JSON object per line. Invalid JSONL raises local `ValueError`; filesystem failures remain SDK I/O errors. SFT rows commonly contain `{"messages": [...]}`; RL datasets may include fields such as `rollout_extra` required by the workflow. The 2.0 purpose default is hyphenated `fine-tune`, not the old `fine_tune` example.

## Fine-Tune Jobs (`client.fine_tunes`)

```text
create(*, model, training_file_ids, validation_file_ids=None,
       mode=None, hyper_parameters=None, finetuned_model=None, ...) -> FineTune
```

- `training_file_ids` and `validation_file_ids` accept a string or list; a string is normalized to a one-element list.
- `finetuned_model` is the request-side custom output name and is sent as wire key `model_name`. It is not the 1.x `finetuned_output` request kwarg.
- `get(job_id)` returns a fresh `FineTune`; `list(page_no=1, page_size=10)` returns `FineTuneList.jobs` and `total_count`.
- `cancel(job_id)` and `delete(job_id)` return flat `JobStatus` snapshots.
- `logs(job_id, offset=1, lines=1000)` returns `JobLogs`; read normalized text from `content`. The Python argument is plural `lines` even though the wire query key is singular.
- `stream_events(job_id)` yields `FineTuneEvent` objects. Event `output` is the event payload, not a legacy response envelope.
- `wait(job_or_job_id, poll_interval=..., max_interval=..., timeout=...)` polls until `SUCCEEDED`, `FAILED`, or `CANCELED`; failed/cancelled jobs raise by default. It returns the final `FineTune` snapshot rather than invoking the deployment automatically.

Flat `FineTune` fields include `job_id`, `job_name`, `status`, `model`, `base_model`, `finetuned_model`, `training_file_ids`, `validation_file_ids`, `hyper_parameters`, `training_type`, `created_at`, `finished_at`, and `usage`, plus common response metadata. `FineTuneList` uses `jobs` and `total_count`.

## Deployments (`client.deployments`)

```text
create(*, model, capacity, model_version=None, suffix=None, plan=None,
       template_id=None, deploy_spec=None, billing_method=None, ...) -> Deployment
```

- `get(deployed_model)` and `scale(deployed_model, *, capacity)` return `Deployment` snapshots. There is no SDK `wait()` on this resource; poll `get()` when readiness is required.
- `list(page_no=1, page_size=10)` returns `DeploymentList.deployments` and `total_count`.
- `delete(deployed_model)` returns `JobStatus`.
- `stream_events(deployed_model_id)` yields `DeploymentEvent`; its `output` is an event payload, not the old response envelope.
- Flat `Deployment` fields include `deployed_model`, `model`, `base_model`, `status`, `capacity`, `ready_capacity`, `charge_type`, `created_at`, and `updated_at`.

The Python SDK's `plan` parameter is optional at the signature level; server-side plan/billing requirements still depend on model and account. This differs from the legacy CLI's required `--plan` option.

## Agentic RL Through `client.rl`

`client.reinforcement_learning` aliases `client.rl`. The resource separates workflow operations from `client.rl.jobs` job management.

### Workflow and function instances

| Method | Contract |
| --- | --- |
| `register_functions(*, rollout_classpaths=None, reward_classpaths=None, group_reward_classpaths=None, workspace_dir="./", lazy_load=True, api_key=None)` | Returns `RegisteredFunctions`, including registered entity IDs and function instance IDs |
| `test_functions(*, instance_id, func_type, input_data, pull_logs=False, log_page_size=100, start_time=None, end_time=None, api_key=None)` | Returns the user function's own result, not a `FineTune` |
| `upload_data(training_files, validation_files=None, *, api_key=None)` | Returns `UploadedData.training_file_ids` and `validation_file_ids` |
| `query_function_instance_logs(...)` | Returns one raw log-query API dictionary |
| `query_all_function_instance_logs(...)` | Returns aggregated log strings |
| `delete_function_instance(...)` | Returns the raw delete API dictionary |

These synchronous workflow methods internally use `asyncio.run`; do not call them from an already running event loop. Use `AsyncClient.rl` there. Their signatures do not accept a generic `timeout=` or per-call `workspace=`; do not copy those options from job methods.

### Jobs (`client.rl.jobs`)

- `create(*, config_path=None, job_name=None, api_key=None, **kwargs) -> FineTune` initializes and runs the complete AgenticRL workflow, then returns after submission. `kwargs` override open-ended training configuration keys. It does not wait for training completion.
- `fetch(job_id) -> FineTune`; `wait(job_or_job_id) -> FineTune`; `list(...) -> FineTuneList`.
- `cancel(job_id)` and `delete(job_id)` return `JobStatus`; `logs(job_id, offset=1, lines=1000)` returns `JobLogs`.

```python
import dashscope

with dashscope.Client() as client:
    submitted = client.rl.jobs.create(config_path="rl_config.yaml")
    job = client.rl.jobs.wait(submitted)
    print(job.job_id, job.status, job.finetuned_model)
```

Use `register_functions()`, `test_functions()`, and `upload_data()` when those stages must be driven individually; do not assume `jobs.create()` reuses objects returned by a separate workflow unless the configuration explicitly wires them together.

## Direct `AgenticRL` Workflow — A Separate Contract

`dashscope.finetune.agentic_rl.AgenticRL` owns the configurable workflow. Typical usage is `AgenticRL(...)`, `init(config_path=..., **kwargs)`, then `await run()`; alternatively drive `register_functions()`, `upload_datasets()`, and synchronous `submit_job()` stage by stage. Components include rollout/reward functions, training/validation datasets, hyperparameters such as `algorithm="gspo"`, and resource settings.

```python
import asyncio
from dashscope.finetune.agentic_rl import AgenticRL

async def main():
    workflow = AgenticRL()
    workflow.init(config_path="rl_config.yaml")
    result = await workflow.run()
    print(result.status_code, result.request_id)

asyncio.run(main())
```

`run()`/`submit_job()` use `SubmitResult`, whose legacy-shaped fields include `status_code`, `request_id`, `code`, `message`, `output`, `usage`, and `headers`. That shape belongs to this workflow API; it is not a `client.rl.jobs` `FineTune`. The public workflow class does **not** provide FineTunes-style `get/list/cancel/delete/logs` methods. For those operations on a submitted RL job, use `client.rl.jobs`.

In 1.x compatibility code, the AgenticRL wrapper adds `cancel/list/get/delete/logs`, and its `submit_job()` returns a 1.x `FineTune`. Keep that wrapper's return and management contract out of current `client.rl` and public `AgenticRL` examples.

## Response and Error Handling

Ordinary 2.0 models expose common `request_id`, `status_code`, `code`, `message`, `is_error`, and `headers` alongside the flat business fields documented above. By default, API/business failures raise exceptions from `dashscope.errors`. If a client is constructed with `raise_on_error=False`, check `response.is_error` before using business fields and retain `request_id` for diagnosis. Intentional raw/event payloads (`Job.output`, `FineTuneEvent.output`, `DeploymentEvent.output`, and direct AgenticRL `SubmitResult.output`) are exceptions, not the standard envelope pattern.

## 1.x Compatibility — Maintenance Only

Use this section only to repair existing imports such as `from dashscope import Files, FineTunes, Deployments`; do not mix these objects with 2.0 client resources.

- `Files.upload(file_path, purpose="fine_tune", description=None)` returned `DashScopeAPIResponse`; old code reads `output["uploaded_files"][0]["file_id"]`. `Files.list` used `page=...`, and invalid fine-tune JSONL raised `InvalidFileFormat`.
- `FineTunes.call(model, training_file_ids, validation_file_ids=None, mode=None, hyper_parameters={}, ...)` created a job. Old custom naming used `kwargs["finetuned_output"]`; management methods were `get/list/cancel/delete/wait/logs/stream_events`.
- Legacy `FineTune.output` fields included `job_id`, `status`, `finetuned_output`, file IDs, hyperparameters, timestamps, and usage. Successful old jobs therefore deployed `response.output.finetuned_output`.
- `Deployments.call(model, capacity, version=None, suffix=None, ...)` created a deployment. Legacy code reads `Deployment.output.deployed_model`, `output.status`, `output.capacity`, and related fields; `scale` used positional capacity.
- Legacy responses generally did not raise for service failures: check `status_code`, then read `code`, `message`, and `request_id`. Do not apply that rule to 2.0 `Client` calls.

## Java SDK

The Java SDK index (v2.22.23) has no FineTunes/Files/Deployments wrappers — for Java, call the Bailian HTTP API directly (endpoints identical to the ones behind the Python calls above).
