#!/usr/bin/env bash
#****************************************************************#
# ScriptName: start_reward_server.sh
# Create Date: 2026-03-17 10:05
# Modify Date: 2026-03-24 17:54
# Function:
#***************************************************************#
set -euo pipefail  # Strict error handling

cd ../

# ================= Configuration ==================
#VERSION="1.25.16"
SERVICE_TYPE="group_reward"                          # group_reward/reward|rollout
PROCESSOR_CLASS="functions.reward.group_reward.DemoGroupRewardProcessor"  # Full class path
PYPI_REPO="https://mirrors.aliyun.com/pypi/simple/"
#SDK_PACKAGE="dashscope-${VERSION}-py3-none-any.whl fastapi pyyaml uvicorn"
REQUIREMENTS_FILE="./requirements.txt"
LOG_DIR="/tmp/log/agentic_rl"
MAX_RETRIES=3

# ================ Helper Functions ================
init_logging() {
    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/service_$(date +%Y%m%d).log"
    exec 3>&1 4>&2
    trap 'exec 1>&3 2>&4' EXIT
    # 关键修复：使用双引号、修正进程替换语法
    exec > >(tee -a "$log_file") 2>&1
    echo -e "\n\n=== Service Start: $(date) ==="
}

log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    log "Cleaning temporary resources..."
    rm -rf "${TMPDIR:-/tmp}/pip*"
    find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
}

validate_environment() {
    log "Validating runtime environment..."

    # Python check
    if ! command -v python3 &>/dev/null; then
        log "ERROR: Python3 not found in PATH"
        exit 101
    fi
}

# ============== Dependency Management ==============
install_with_retry() {
    local packages=("$@")
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        log "Installing ${packages[*]} (attempt $((retry_count+1))/${MAX_RETRIES})"
        if python3 -m pip install -U "${packages[@]}" \
            --index-url "${PYPI_REPO}" \
            --no-cache-dir \
            --compile; then
            return 0
        fi
        retry_count=$((retry_count+1))
        sleep $((retry_count * 10))
    done

    log "FATAL: Failed to install ${packages[*]} after ${MAX_RETRIES} attempts"
    return 1
}

# ================ Main Execution ===================
main() {
    # Phase 1: Initialization
    init_logging
    trap cleanup EXIT
    validate_environment

    # Phase 2: Dependency Setup
    log "Starting dependency installation"

#    local packages=($SDK_PACKAGE)
#    for pkg in "${packages[@]}"; do
#      if ! install_with_retry "$pkg"; then
#          exit 201
#      fi
#    done

    if [ -f "${REQUIREMENTS_FILE}" ]; then
        log "Installing additional requirements"
        if ! install_with_retry -r "${REQUIREMENTS_FILE}"; then
            exit 202
        fi
    fi

    # Phase 3: Environment Configuration
    export FUNC_TYPE="${SERVICE_TYPE}"
    export PROCESSOR_CLASS="${PROCESSOR_CLASS}"
    export PYTHONPATH=".:${PYTHONPATH:-}"

    log "Final Environment:"
    env | grep -E 'FUNC_TYPE|PROCESSOR_CLASS|PYTHONPATH'

    # Phase 4: Service Launch
    log "Starting ${SERVICE_TYPE} service"
    exec python3 -m dashscope.finetune.reinforcement.component.server.server "$@"
}

# ==================== Entry ========================
main "$@"
