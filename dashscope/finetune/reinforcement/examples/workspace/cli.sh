#!/usr/bin/env bash
#****************************************************************#
# ScriptName: cli.sh
# Create Date: 2026-03-04 17:22
# Modify Date: 2026-03-25 14:27
# Function:
#***************************************************************#

#Environment Variables Configuration:
#--------------------------------------------------------------
#1. DASHSCOPE_API_KEY (Required)
#   - Sets the authentication key for DashScope API access
#   - Format: API key string (e.g., "ds-1234567890abcdef")
#
#2. LOG_LEVEL (Optional, default: 'info')
#   - Controls logging verbosity
#   - Allowed values: 'debug', 'info', 'warning', 'error', 'critical'
#
#3. FC_PYPI_LIB (Optional)
#   - Specifies local/alternative dashscope package installations
#   - Accepts space-separated list of wheel paths
#   - Default: Uses official PyPI 'dashscope' package
#   - Example: "./dist/dashscope-1.3.0-py3-none-any.whl ./custom_deps/custom_lib.whl"
#
#CLI Command Documentation:
#--------------------------------------------------------------
#1. Register Function Components
#   - Command: `dashscope rl register_functions`
#   - Function: Registers RL workflow components
#   - Key Options:
#     * --rollout-classpaths: Classpaths for rollout processors
#     * --reward-classpaths: Classpaths for reward calculators
#     * --group-reward-classpaths: Classpaths for group reward processors
#     * --no-lazy-load: Immediate component initialization
#   - Outputs: JSON with entity/instance IDs for registered components
#   - Example:
#     ```bash
#     dashscope rl register_functions \
#       --rollout-classpaths "rollout.Processor" \
#       --reward-classpaths "reward.Calculator" \
#       -o json
#     ```
#
#2. Test Component Functions
#   - Command: `dashscope rl test_functions <INSTANCE_ID>`
#   - Function: Validates component functionality
#   - Key Options:
#     * --type: Component type (rollout/reward/group_reward)
#     * --input: Test input JSON file path
#   - Test Types:
#     a. Rollout Test:
#        ```bash
#        dashscope rl test_functions "ro-ins-123" \
#          --type rollout \
#          --input rollout_test.json
#        ```
#     b. Reward Test:
#        ```bash
#        dashscope rl test_functions "rw-ins-456" \
#          --type reward \
#          --input reward_test.json
#        ```
#     c. Group Reward Test:
#        ```bash
#        dashscope rl test_functions "grw-ins-789" \
#          --type group_reward \
#          --input group_reward_test.json
#        ```
#
#3. Data Management
#   - Command: `dashscope rl upload_data`
#   - Function: Uploads training datasets
#   - Key Options:
#     * --training-files: Paths to training datasets
#     * --validation-files: Paths to validation datasets
#   - Output: Uploaded dataset IDs
#   - Example:
#     ```bash
#     dashscope rl upload_data \
#       --training-files "./data/train.jsonl" \
#       -o json
#     ```
#
#4. Run Complete Workflow
#   - Command: `dashscope rl run`
#   - Function: End-to-end RL job submission and execution
#   - Operation Modes:
#     a. Config-Driven Mode:
#        ```bash
#        dashscope rl run -c job.yaml -o json
#        ```
#   - Key Options:
#     * -c/--config: YAML configuration file path
#     * --verbose: Show detailed execution logs
#
#5. Job Lifecycle Management
#   - Status Check:
#     ```bash
#     dashscope rl get <JOB_ID> -o json
#     ```
#   - Log Retrieval:
#     ```bash
#     dashscope rl logs <JOB_ID> --lines 50
#     ```
#   - Job Listing:
#     ```bash
#     dashscope rl list --page 1 --size 10
#     ```
#   - Job Termination:
#     ```bash
#     dashscope rl cancel <JOB_ID>
#     ```
#   - Job Deletion:
#     ```bash
#     dashscope rl delete <JOB_ID>
#     ```
#
#Output Formats:
#------------------------------------------------------------------------
#| Option       | Description                  | Use Case               |
#|--------------|------------------------------|------------------------|
#| -o json      | Machine-readable JSON        | Automation pipelines   |
#| (default)    | Human-friendly table format  | Interactive debugging  |
#
#All commands support --help for detailed parameter information.

set -e

# Helper: extract JSON field using Python (no jq dependency)
json_get() { python3 -c "import sys,json; print(json.loads(sys.stdin.read())$1)"; }
json_list() { python3 -c "import sys,json; [print(x) for x in json.loads(sys.stdin.read())$1]"; }

# ===================== 1. Register functions =====================
echo ">>> Step 1: Registering function components..."
REGISTER_RESULT=$(dashscope rl register_functions \
  --rollout-classpaths "functions.rollout.rollout_only.DemoRolloutProcessor" \
  --reward-classpaths "functions.reward.reward.DemoRewardProcessor" \
  --reward-classpaths "functions/reward/reward_decorator.py:SafetyProcessor" \
  --group-reward-classpaths "functions.reward.group_reward.DemoGroupRewardProcessor" \
  --no-lazy-load \
  --output-format json)

echo "$REGISTER_RESULT"

ROLLOUT_INSTANCE_ID=$(echo "$REGISTER_RESULT" | json_get "['rollout_instance_ids'][0]")
REWARD_INSTANCE_IDS=($(echo "$REGISTER_RESULT" | json_list "['reward_instance_ids']"))
GROUP_REWARD_INSTANCE_ID=$(echo "$REGISTER_RESULT" | json_get "['group_reward_instance_ids'][0]")

# ===================== 2. Test functions =====================
echo ">>> Step 2: Testing rollout function (instance: $ROLLOUT_INSTANCE_ID)..."
dashscope rl test_functions "$ROLLOUT_INSTANCE_ID" \
  --type rollout \
  --input ./resources/rollout_input.json

echo ">>> Step 2: Testing reward function (instance: ${REWARD_INSTANCE_IDS[0]})..."
dashscope rl test_functions "${REWARD_INSTANCE_IDS[0]}" \
  --type reward \
  --input ./resources/reward_input.json

echo ">>> Step 2: Testing reward function (instance: ${REWARD_INSTANCE_IDS[1]})..."
dashscope rl test_functions "${REWARD_INSTANCE_IDS[1]}" \
  --type reward \
  --input ./resources/reward_decorator_input.json

echo ">>> Step 2: Testing group_reward function (instance: $GROUP_REWARD_INSTANCE_ID)..."
dashscope rl test_functions "$GROUP_REWARD_INSTANCE_ID" \
  --type group_reward \
  --input ./resources/group_reward_input.json

# ===================== 3. Upload dataset =====================
echo ">>> Step 3: Uploading datasets..."
UPLOAD_RESULT=$(dashscope rl upload_data \
  --training-files "./data/calc_train_min.jsonl" \
  --validation-files "./data/calc_validation_min.jsonl" \
  -o json)
echo "$UPLOAD_RESULT"

# ===================== 4. Submit job =====================
echo ">>> Step 4: Submitting job..."
RUN_RESULT=$(dashscope rl run -c job.yaml -o json)
echo "$RUN_RESULT"

JOB_ID=$(echo "$RUN_RESULT" | json_get "['job_id']")

# ===================== 5. Job lifecycle =====================
echo ">>> Step 5: Checking job status (job: $JOB_ID)..."
dashscope rl get "$JOB_ID" -o json

echo ">>> Step 5: Fetching job logs..."
dashscope rl logs "$JOB_ID" --offset 1 --lines 50

#echo ">>> Listing jobs..."
#dashscope rl list --page 1 --size 10

#echo ">>> Canceling job..."
#dashscope rl cancel "$JOB_ID"

#echo ">>> Deleting job..."
#dashscope rl delete "$JOB_ID"
