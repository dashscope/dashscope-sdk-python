#!/usr/bin/env bash

set -euo pipefail  # Strict error handling

# ================= Configuration ==================
SERVICE_TYPE="reward"                          # reward|rollout
PROCESSOR_CLASS="functions.reward.reward.DemoRewardProcessor"  # Full class path
PYPI_REPO="https://mirrors.aliyun.com/pypi/simple/"
SDK_PACKAGE=""
REQUIREMENTS_FILE="./requirements.txt"
SERVER_CLASSPATH="dashscope.finetune.reinforcement.component.server.server"
WORKERS_COUNT="2"
FUNCTION_LAYER="True"
LOG_DIR="/tmp/log/agentic_rl"
MAX_RETRIES=3

# ================ Helper Functions ================
init_logging() {
    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/service_$(date +%Y%m%d).log"
    exec 3>&1 4>&2
    trap 'exec 1>&3 2>&4' EXIT
    exec > >(tee -a "$log_file") 2>&1
    echo -e "\n\n=== Service Start: $(date) ==="
}

log() {
    printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    log "Cleaning temporary workspace..."
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
        log "Installing ${packages[*]} ($((retry_count+1))/${MAX_RETRIES})"
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

    if [ "${FUNCTION_LAYER}" = "False" ]; then
        # Phase 2:
        if ! install_with_retry "virtualenv"; then
            log "Failed to install default package: virtualenv"
            exit 202
        fi
        virtualenv dashscope-env
        source dashscope-env/bin/activate
    fi

    # Phase 3: Default dependency Setup
    log "Installing default packages"
    local_packages=($SDK_PACKAGE)
    for pkg in "${local_packages[@]}"; do
        if ! install_with_retry "$pkg"; then
            log "Failed to install default package: $pkg"
            exit 203
        fi
    done

    if [ "${FUNCTION_LAYER}" = "False" ]; then
        # Phase 4: User dependency Setup
        log "Starting user dependency installation"
        if [ -f "${REQUIREMENTS_FILE}" ]; then # Check if file exists
            log "Installing additional requirements from ${REQUIREMENTS_FILE}"
            if ! install_with_retry -r "${REQUIREMENTS_FILE}"; then
                log "Failed to install requirements from ${REQUIREMENTS_FILE}"
                exit 204
            fi
        fi
    fi

    # Phase 5: Environment Configuration
    export FUNC_TYPE="${SERVICE_TYPE}"
    export PROCESSOR_CLASS="${PROCESSOR_CLASS}"
    export PYTHONPATH=".:${PYTHONPATH:-}"
    export WORKERS_COUNT="${WORKERS_COUNT}"

    log "Final Environment:"
    env | grep -E 'FUNC_TYPE|PROCESSOR_CLASS|PYTHONPATH'

    # Phase 6: Service Launch
    log "Starting ${SERVICE_TYPE} service"
    exec python3 -m "${SERVER_CLASSPATH}" "$@"
}
# ==================== Entry ======================
main "$@"
