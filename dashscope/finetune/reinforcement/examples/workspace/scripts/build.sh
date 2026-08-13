#!/usr/bin/env bash
#****************************************************************#
# ScriptName: build.sh
# Create Date: 2026-03-04 17:22
# Modify Date: 2026-03-25 14:27
# Function:
#   Build and optionally install DashScope SDK for Agentic RL development
#   Default behavior: Build wheel only, no installation
#   Set INSTALL_LOCAL=1 to install local build
#   Automatically detects the latest built wheel file
#***************************************************************#

# Configuration
WORK_DIR='dashscope/finetune/reinforcement/examples/workspace/'

# Parse command line arguments
INSTALL_FLAG=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--install)
            INSTALL_FLAG=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find project root directory (where setup.py is located)
find_project_root() {
    # Start from current directory
    local current_dir="$PWD"

    # Look for setup.py in current and parent directories
    while [[ "$current_dir" != "/" ]]; do
        if [[ -f "$current_dir/setup.py" ]]; then
            echo "$current_dir"
            return 0
        fi
        current_dir=$(dirname "$current_dir")
    done

    echo "Error: Could not find project root (setup.py)" >&2
    return 1
}

# Build SDK wheel function
build_sdk_wheel() {
    echo "Building SDK wheel..."
    if ! python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1; then
        echo "Error: Failed to upgrade build tools" >&2
        return 1
    fi

    if ! python3 setup.py sdist bdist_wheel >/dev/null 2>&1; then
        echo "Error: Failed to build wheel package" >&2
        return 1
    fi

    rm -rf build
    return 0
}

# Find the latest wheel file
find_latest_wheel() {
    wheels=($(find dist -name "dashscope-*.whl" 2>/dev/null))

    if [ ${#wheels[@]} -eq 0 ]; then
        echo "Error: No wheel files found in dist directory" >&2
        return 1
    fi

    # Sort by modification time (newest first)
    latest=$(ls -t dist/*.whl | head -n1 2>/dev/null)

    if [ -z "$latest" ]; then
        echo "Error: Failed to determine latest wheel" >&2
        return 1
    fi

    echo "$latest"
    return 0
}

# Install local wheel function
install_local_wheel() {
    echo "Installing local wheel..."
    if ! pip3 uninstall -y dashscope >/dev/null 2>&1; then
        echo "Warning: Failed to uninstall existing dashscope" >&2
    fi

    if ! pip3 install -U "$1" -i https://mirrors.aliyun.com/pypi/simple/ >/dev/null 2>&1; then
        echo "Error: Failed to install local wheel" >&2
        return 1
    fi

    return 0
}

# Main script execution
main() {
    # Find project root
    PROJECT_ROOT=$(find_project_root) || exit 1
    echo "Project root: $PROJECT_ROOT"

    # Navigate to project root
    pushd "$PROJECT_ROOT" >/dev/null || exit 1

    # Build SDK wheel
    if ! build_sdk_wheel; then
        echo "Build failed" >&2
        popd >/dev/null
        exit 1
    fi

    # Find the latest wheel file
    LATEST_WHL=$(find_latest_wheel)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to find wheel file" >&2
        popd >/dev/null
        exit 1
    fi

    # Extract the wheel filename
    WHL_FILENAME=$(basename "$LATEST_WHL")
    echo "Latest wheel file: $WHL_FILENAME"

    # Create workspace directory if needed
    WORKSPACE_PATH="${PROJECT_ROOT}/${WORK_DIR}"
    if [ ! -d "$WORKSPACE_PATH" ]; then
        mkdir -p "$WORKSPACE_PATH"
    fi

    # Copy wheel file to workspace
    echo "Copying wheel ($WHL_FILENAME) to workspace..."
    if ! cp -v "$LATEST_WHL" "${WORKSPACE_PATH}/"; then
        echo "Error: Failed to copy wheel to workspace" >&2
        popd >/dev/null
        exit 1
    fi

    # Install local wheel if requested
    if [ "$INSTALL_FLAG" = "1" ]; then
        if ! install_local_wheel "${WORKSPACE_PATH}/${WHL_FILENAME}"; then
            echo "Installation failed" >&2
            popd >/dev/null
            exit 1
        fi
        echo "Successfully installed local build"
    else
        echo "Local build created but not installed (use --install to install)"
    fi

    # Return to original directory
    popd >/dev/null
    exit 0
}

# Execute main function
main
