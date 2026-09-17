# -*- coding: utf-8 -*-
"""
Environment Variables Configuration:
--------------------------------------------------------------
1. DASHSCOPE_API_KEY (Required)
   - Sets the authentication key for DashScope API access
   - Format: API key string (e.g., "sk-***")

2. LOG_LEVEL (Optional, default: 'info')
   - Controls logging verbosity
   - Allowed values: 'debug', 'info', 'warning', 'error', 'critical'

3. FC_PYPI_LIB (Required)
   - Specifies local/alternative dashscope package installations
   - Accepts space-separated list of wheel paths
   - Default: Uses official PyPI 'dashscope' package
   - Example: "dashscope-1.25.16-py3-none-any.whl"

Method Documentation:
--------------------------------------------------------------
1. init(config_path: str, **kwargs)
   - Initializes the RL workflow using a YAML configuration file
   - Key Parameters:
     * config_path (required): Path to YAML configuration file (e.g.,
     "rl-job.yaml")
     * job_name: Custom identifier for the training job
   - Configuration File Example (rl-job.yaml)
   - Benefits:
     * Decouples configuration from code
     * Enables version-controlled parameter management
     * Supports environment-specific configurations

2. run(**kwargs)
   - Executes the RL training workflow with current configuration
   - When used with init():
     * Requires no arguments - uses preloaded YAML configuration
     * Optional: lazy_load=False forces immediate component validation
   - When used without init():
     * Requires explicit parameters:
       - model: Base model ID
       - training_files/validation_files: Dataset paths
       - functions: List of component configurations
       - hyper_parameters: Training settings
   - Returns: Job submission result with job_id
"""

# Example YAML-driven usage:
# client.init(config_path="rl-job.yaml", job_name="agentic-rl")
# await client.run()  # Uses YAML configuration

from dashscope.finetune.agentic_rl import AgenticRL
from dashscope.finetune.reinforcement import (
    logger,
    DataSourceType,
    TrainingDataset,
    ValidationDataset,
    RolloutFunctionComponent,
    RewardFunctionComponent,
    FunctionComponentModel,
    FunctionComponentRuntime,
)


async def main_workflow():
    """Main execution workflow"""
    try:
        logger.info("Starting main tests：workflow")

        # Defines infrastructure specs for rollout components
        # env: inject global variables or environment-specific
        # configurations into function components. example: "env":{
        # "ENABLE_TRAJECTORY": True}
        rollout_runtime = {
            "cpu": 2,
            "memory_size": 4096,
            "disk_size": 512,
            "concurrency": 30,
            "capacity": 30,
            "min_capacity": 30,
            "max_capacity": 60,
            "memory_scale_threshold": 0.6,
            "concurrency_scale_threshold": 0.6,
            "env": {},
        }
        reward_runtime = FunctionComponentRuntime(
            cpu=2,
            memory_size=4096,
            disk_size=512,
            concurrency=30,
            env={},
            capacity=30,
            min_capacity=30,
            max_capacity=60,
            memory_scale_threshold=0.6,
            concurrency_scale_threshold=0.6,
        )

        # Main client for agentic reinforcement learning workflows
        client = AgenticRL()  # client = AgenticRL(api_key="sk-***")
        result = await client.run(
            job_name="agentic-rl",
            model="qwen3.5-9b",
            # Datasets Configuration
            # Specifies training data (calc_train_min.jsonl) and validation
            # data (calc_validation_min.jsonl) paths
            training_datasets=[
                TrainingDataset(
                    data_source_type=DataSourceType.FILE_ID,
                    file_name="./data/calc_train_min.jsonl",
                ),
            ],
            validation_datasets=[
                ValidationDataset(
                    data_source_type=DataSourceType.FILE_ID,
                    file_name="./data/calc_validation_min.jsonl",
                ),
            ],
            # Function Component Configuration
            functions=[
                # RolloutFunctionComponent：Environment simulation/rollout
                # generation
                RolloutFunctionComponent(
                    name="rollout-1",
                    timeout=600,
                    fcmodel=FunctionComponentModel(
                        classpath="functions.rollout.rollout.CalcXRolloutProcessor",  # noqa: E501
                    ),
                    runtime=FunctionComponentRuntime(**rollout_runtime),
                ),
                # RewardFunctionComponent：Reward calculation engine
                RewardFunctionComponent(
                    name="reward-1",
                    weight=1.0,
                    timeout=120,
                    reward_metric_weight={
                        "reward_metric_weightA": 0.3,
                        "reward_metric_weightB": 0.7,
                    },
                    fcmodel=FunctionComponentModel(
                        classpath="functions.reward.reward.DemoRewardProcessor",  # noqa: E501
                    ),
                    runtime=reward_runtime,
                ),
            ],
            # Training Configuration
            hyper_parameters={
                # Policy optimization algorithm (Generalized Supervised
                # Policy Optimization)
                "algorithm": "gspo",
                "batch_size": 64,  # Training samples per optimization step
                "eval_steps": 1,  # Run evaluation every N training steps
                # Weight for KL divergence loss (prevents policy divergence)
                "kl_loss_coef": 0.002,
                # Initial step size for gradient updates (fine-tuning)
                "learning_rate": 2e-6,
                "lr_scheduler_type": "linear",  # Learning rate decay strategy
                "max_length": 8192,  # Max sequence length for model input
                "n_epochs": 1,  # Full passes through training data
                "n_rollouts": 8,  # Parallel environment rollouts per batch
                "ppo_mini_batch_size": 8,  # Samples per PPO optimization sub-step  # noqa: E501  # pylint: disable=line-too-long
                "save_strategy": "steps",  # Model checkpoint frequency: 'steps'  # noqa: E501  # pylint: disable=line-too-long
            },
            # Cloud resource specifications
            resources={
                "charge_type": "mtu_postpaid",
                "mtu_spec_code": "MTU4",
                "mtu_capacity": 24,
            },
        )

        if result.status_code == 200:
            job_id = result.output.job_id
            logger.info(f"agentic rl submit: {job_id=}, {result=}")
        else:
            raise ValueError(f"agentic rl submit: {result}")

        # Get rl job
        result = AgenticRL.get(job_id=job_id)
        status = result.status_code
        logger.info(f"agentic rl get: {job_id=}, {status=}, {result=}")

        # View training logs
        # Please check them in the Bailian Model Tuning Console

        # Cancel rl job
        # result = AgenticRL.cancel(job_id=job_id)
        # status = result.status_code
        # logger.info(f"agentic rl cancel: {job_id=}, {status=}, {result=}")

        logger.info("All tests workflows completed successfully")

    except Exception as e:
        root = e.root_cause if hasattr(e, "root_cause") else e
        logger.error(f"Main execution flow terminated: {root}")


async def main_workflow_yaml():
    # Configuration-Driven (Recommended)
    try:
        logger.info("Starting main tests：workflow from yaml")

        client = AgenticRL()
        client.init(
            config_path="rl-job.yaml",
            name="agentic-rl-from-yaml",
        )
        client.tuning.to_yaml(file_path="init.yaml")

        # Submit job
        result = (
            await client.run()
        )  # Automatically reads cached configuration, no arguments required
        if result.status_code == 200:
            job_id = result.output.job_id
            logger.info(
                f"agentic rl(from rl-job.yaml) submit: {job_id=}, {result=}",
            )
        else:
            raise ValueError(f"agentic rl submit: {result}")

        # Get rl job
        result = AgenticRL.get(job_id=job_id)
        status = result.status_code
        logger.info(f"agentic rl get: {job_id=}, {status=}, {result=}")

    except Exception as e:
        root = e.root_cause if hasattr(e, "root_cause") else e
        logger.error(f"Main execution flow terminated: {root}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_workflow())
    asyncio.run(main_workflow_yaml())
