# -*- coding: utf-8 -*-
import ast
import asyncio
import copy
import fnmatch
import json
import os
import uuid
import zipfile
from typing import Optional, List, Any, Dict, Union, Tuple, Literal

import aiohttp
import requests
from aiohttp import FormData
from tenacity import (
    retry,
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from dashscope.finetune.reinforcement import logger
from dashscope.finetune.reinforcement.common.errors import (
    InputError,
    OutputError,
    ConfigurationError,
    BasePermissionError,
    RuntimeErrorWithCode,
    OSSUploadError,
)
from dashscope.finetune.reinforcement import (
    LOG_LEVEL,
    DASHSCOPE_API_KEY,
    BAILIAN_FILE_API,
    BAILIAN_FILE_TIMEOUT,
    HTTP_REQUEST_TIMEOUT,
    FC_FILES_START,
    FC_PYPI_LIB,
    FC_PYPI_REPO,
    FC_LAYER_USED,
    FC_SERVER_CLASSPATH,
    FC_ZIP_EXCLUDE_PATTERNS,
    FC_OSS_FILE_SIZE_WARNING,
    DATASETS_FILE_SIZE_WARNING,
    LOGGER_FILTER_FIELDS,
    FC_WORKERS_COUNT,
)
from dashscope.finetune.reinforcement.common.model_types import (
    FileSpec,
    FunctionType,
)


def generate_random_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    uuid4 = uuid.uuid4()
    return f"{prefix}-{uuid4}" if prefix else str(uuid4)


async def async_http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Union[Dict[str, Any], FormData]] = None,
    timeout: int = 10,
    retry_times: int = 3,
) -> Dict[str, Any]:
    """Perform an asynchronous HTTP request with tenacity retries."""

    async def _make_request() -> Dict[str, Any]:
        """Single attempt: return result on success, raise exception on
        failure."""
        async with aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            trust_env=True,
        ) as session:
            method_upper = method.upper()

            if method_upper == "GET":
                async with session.get(url, params=data) as response:
                    result = await _handle_response(response)
            elif method_upper == "POST":
                async with session.post(url, json=data) as response:
                    result = await _handle_response(response)
            elif method_upper == "POST-DATA":
                async with session.post(url, data=data) as response:
                    result = await _handle_response(response)
            else:
                # Logical error – do not retry
                raise InputError(
                    f"Unsupported method: {method}",
                    error_code=4001,
                )

            # Treat server 5xx responses as transient failures and trigger a
            # retry
            if (
                isinstance(result.get("status"), Dict)
                and result.get("status").get("code") >= 500
            ):
                raise aiohttp.ClientError(
                    f"Server error {result['status']['code']}:"
                    f" {result['status']['message']}",
                )
            return result

    # Retry only on network errors, timeouts, and 5xx (wrapped as ClientError)
    retryer = AsyncRetrying(
        stop=stop_after_attempt(retry_times),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=10,
        ),  # exponential backoff
        retry=retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError),
        ),
        reraise=True,  # raise RetryError when exhausted
    )

    try:
        async for attempt in retryer:
            with attempt:
                return await _make_request()
    except InputError:
        raise
    except asyncio.TimeoutError as e:
        raise RuntimeErrorWithCode(
            f"Request timeout ({timeout}s)",
            error_code=4003,
        ) from e
    except aiohttp.ClientError as e:
        raise RuntimeErrorWithCode(
            f"Client error: {e}",
            error_code=4002,
        ) from e
    except Exception as e:
        raise RuntimeErrorWithCode(
            f"Unexpected error: {e}",
            error_code=4004,
        ) from e

    raise RuntimeErrorWithCode("Unexpected control flow", error_code=4005)


async def _handle_response(response) -> Dict[str, Any]:
    """Handle HTTP response and extract JSON data."""
    try:
        content = await response.json()
        content.setdefault(
            "status",
            {"code": response.status, "message": response.reason},
        )
        return content
    except json.JSONDecodeError:
        return {
            "status": {"code": 4004, "message": "Invalid JSON response"},
            "output": await response.text(),
        }


async def client_fc(
    api_key: str,
    url: str,
    request_data: Dict,
    method: str = "POST",
    content_type: str = "application/json",
) -> Dict:
    """Client function for Function Compute API requests."""
    return await async_http_request(
        method=method,
        url=url,
        headers={
            "Content-Type": content_type,
            "Authorization": "Bearer " + api_key,
        },
        data=request_data,
        timeout=HTTP_REQUEST_TIMEOUT,
    )


def check_file(file: str) -> None:
    """Validate file existence and accessibility."""
    if not os.path.exists(file):
        raise InputError(f"File {file} not found", error_code=4011)
    if not os.path.isfile(file):
        raise InputError(f"{file} is not a file", error_code=4012)
    if not os.access(file, os.R_OK):
        raise InputError(
            f"No read access to file: {file}",
            error_code=4013,
        )


def generate_agentic_script(
    fc_pypi_lib: str,
    fc_pypi_repo: str,
    func_type: str,
    classpath: str,
    requirements_path: Optional[str] = None,
    function_layer_used: bool = True,
) -> str:
    """
    Generate robust deployment script with error handling.

    Args:
        fc_pypi_repo: PyPI repository URL
        requirements_path: Path to requirements.txt
        func_type: Function type (reward/rollout)
        classpath: Full processor class path

    Returns:
        Generated bash script content
    """
    shell_script_header = f"""#!/usr/bin/env bash

set -euo pipefail  # Strict error handling

# ================= Configuration ==================
SERVICE_TYPE="{func_type}"                          # reward|rollout
PROCESSOR_CLASS="{classpath}"  # Full class path
PYPI_REPO="{fc_pypi_repo}"
SDK_PACKAGE="{fc_pypi_lib}"
REQUIREMENTS_FILE="{requirements_path}"
SERVER_CLASSPATH="{FC_SERVER_CLASSPATH}"
WORKERS_COUNT="{FC_WORKERS_COUNT}"
FUNCTION_LAYER="{function_layer_used}"
LOG_DIR="/tmp/log/agentic_rl"
MAX_RETRIES=3
"""

    shell_script_content = r"""
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
"""

    shell_script_main = """# ==================== Entry ======================
main "$@"
"""

    return shell_script_header + shell_script_content + shell_script_main


def create_deployment_files(
    functype: FunctionType,
    dirpath: str,
    filepath: str,
    classname: str,
    requirements_path: Optional[str] = None,
) -> None:
    """Create startup script and requirements file for deployment."""
    try:
        # Validate main Python file
        path = os.path.join(dirpath, filepath)
        check_file(path)
        logger.debug(f"Found code file: {path}")

        # Validate requirements file if provided
        if requirements_path and requirements_path.strip():
            req_path = os.path.join(dirpath, requirements_path)
            check_file(req_path)
            logger.debug(f"Found requirements file: {req_path}")

        # Create startup script
        classpath = (
            os.path.normpath(filepath).replace("/", ".").rsplit(".py", 1)[0]
            + "."
            + classname
        )
        content = generate_agentic_script(
            fc_pypi_lib=FC_PYPI_LIB,
            fc_pypi_repo=FC_PYPI_REPO,
            func_type=str(functype),
            classpath=classpath,
            requirements_path=requirements_path,
            function_layer_used=FC_LAYER_USED,
        )

        with open(FC_FILES_START, "w", encoding="utf-8") as f:
            f.write(content)

        # Add execute permission on Unix systems
        if os.name == "posix":
            os.chmod(FC_FILES_START, 0o755)

        logger.debug(f"Generated startup script: {FC_FILES_START}")
    except Exception as e:
        raise RuntimeErrorWithCode(
            "Deployment file creation error",
            error_code=4021,
        ) from e


def zip_files(files: List[str], output_zip: str) -> None:
    """Zip multiple files into a single archive."""
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file)


# pylint: disable=too-many-branches
def zip_dir(
    dirpath: str,
    output_zip: str,
    extra_files: Optional[List[str]] = None,
    rw_type: Literal["r", "w", "x", "a"] = "w",
    exclude_patterns: Optional[List[str]] = None,
) -> None:
    """
    Compress a directory and optional extra files, with exclusion support.

    Args:
        dirpath: Main directory path to compress
        output_zip: Output zip file path
        extra_files: List of additional files to add
        rw_type: Zip file write mode ('w', 'a', etc.)
        exclude_patterns: List of patterns to exclude (e.g. ["*.log",
        "__pycache__"])
    """

    def _should_exclude(path: str, patterns: List[str]) -> Optional[str]:
        """Return matched pattern if path should be excluded,
        None otherwise."""
        for pattern in patterns:
            if fnmatch.fnmatch(
                os.path.basename(path),
                pattern,
            ) or fnmatch.fnmatch(path, pattern):
                return pattern
        return None

    if exclude_patterns is None:
        env_exclude = FC_ZIP_EXCLUDE_PATTERNS
        exclude_patterns = [
            p.strip() for p in env_exclude.split(",") if p.strip()
        ]

    all_excludes = exclude_patterns
    logger.debug(f"Zip exclusion patterns: {all_excludes}")

    try:
        with zipfile.ZipFile(
            output_zip,
            rw_type,
            zipfile.ZIP_DEFLATED,
        ) as zipf:
            # Compress main directory
            if os.path.exists(dirpath):
                for root, dirs, files in os.walk(dirpath, topdown=True):
                    the_dirs = []
                    for d in dirs:
                        full_rel_path = os.path.join(
                            os.path.relpath(root, start=dirpath),
                            d,
                        )
                        normalized_path = full_rel_path.replace("\\", "/")

                        matched = _should_exclude(
                            normalized_path,
                            all_excludes,
                        )

                        if not matched:
                            the_dirs.append(d)
                        else:
                            logger.debug(
                                f"Excluding directory: {normalized_path} ("
                                f"matched pattern: '{matched}')",
                            )
                            continue

                    dirs[:] = the_dirs

                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, start=dirpath)

                        if any(
                            fnmatch.fnmatch(os.path.basename(file), pattern)
                            or fnmatch.fnmatch(file, pattern)
                            for pattern in all_excludes
                        ):
                            logger.debug(f"Excluding file: {rel_path}")
                            continue

                        zipf.write(file_path, rel_path)
            else:
                logger.warning(f"Directory not found: {dirpath}")

            # Add extra files to zip root directory
            if extra_files:
                for file in extra_files:
                    if os.path.exists(file):
                        if any(
                            fnmatch.fnmatch(os.path.basename(file), pattern)
                            for pattern in all_excludes
                        ):
                            logger.debug(f"Excluding extra file: {file}")
                            continue

                        zipf.write(file, os.path.basename(file))
                    else:
                        logger.warning(f"Extra file not found: {file}")

    except Exception as e:
        raise RuntimeErrorWithCode(
            "Directory compression error",
            error_code=4022,
        ) from e


def _sync_upload_to_oss(signed_url: str, zip_path: str) -> int:
    """Synchronously upload a file to OSS with progress tracking."""
    try:
        file_size = os.path.getsize(zip_path)
        size_mb = file_size / (1024 * 1024)
        if file_size > FC_OSS_FILE_SIZE_WARNING:
            logger.warning(
                f"Uploading large file: {zip_path} ({size_mb:.2f}MB) to OSS",
            )
            raise OSSUploadError(
                f"Uploading large file: {zip_path} ({size_mb:.2f}MB) to OSS",
            )

        logger.debug(
            f"Uploading file: {zip_path} ({size_mb:.2f}MB) to OSS",
        )

        with open(zip_path, "rb") as file:
            response = requests.put(
                signed_url,
                data=file,
                headers={},
                timeout=BAILIAN_FILE_TIMEOUT,
            )

            if response.status_code != 200:
                error_msg = response.text
                raise OSError(
                    f"OSS upload failed ({response.status_code}): {error_msg}",
                )

            return response.status_code
    except Exception as e:
        raise RuntimeErrorWithCode(
            "OSS upload error",
            error_code=4023,
        ) from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ),
    reraise=True,
)
async def upload_zip_to_oss_and_by_signed_url(
    signed_url: str,
    zip_path: str,
) -> int:
    """Asynchronously upload ZIP file to OSS with retry mechanism."""
    try:
        return await asyncio.to_thread(
            _sync_upload_to_oss,
            signed_url,
            zip_path,
        )
    except BasePermissionError:
        raise  # Re-raise permission errors directly
    except Exception as e:
        if "403" in str(e):
            raise BasePermissionError(
                "OSS access denied (403)",
                error_code=4024,
            ) from e
        raise


async def to_bailian_data(files: List[FileSpec]) -> List[str]:
    """
    Upload files to Bailian file storage service.

    Returns:
        List of uploaded file IDs

    Raises:
        OutputError: If file upload fails
    """
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    form_data = FormData()
    uploaded_files = []

    try:
        valid_file_count = 0
        for file_spec in files:
            file_path = file_spec.path
            descriptions = file_spec.descriptions

            # Validate file
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
            if not os.path.isfile(file_path):
                logger.error(f"Not a file: {file_path}")
                continue
            if os.path.getsize(file_path) == 0:
                logger.error(f"Empty file: {file_path}")
                continue

            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            if file_size > DATASETS_FILE_SIZE_WARNING:
                raise InputError(
                    f"File too large: {file_path} ({size_mb:.1f}MB), "
                    f"max allowed "
                    f"{DATASETS_FILE_SIZE_WARNING / (1024 * 1024):.0f}MB. "
                    f"Adjust via env var DATASETS_FILE_SIZE_WARNING.",
                    error_code=4035,
                )

            # Add file to form data
            with open(file_path, "rb") as f:
                form_data.add_field(
                    name="files",
                    value=f.read(),
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )

            form_data.add_field("purpose", "fine-tune")
            if descriptions:
                form_data.add_field("descriptions", descriptions)

            valid_file_count += 1

        # Check if there are any valid files to upload
        if valid_file_count == 0:
            raise InputError(
                "No valid files found to upload. All files failed validation.",
                error_code=4031,
            )

        # Execute upload request
        result = await async_http_request(
            method="POST-DATA",
            url=BAILIAN_FILE_API,
            headers=headers,
            data=form_data,
            timeout=BAILIAN_FILE_TIMEOUT,
            retry_times=1,
        )

        # Handle errors
        if result.get("status", {}).get("code", 200) != 200:
            raise OutputError(
                f"File upload failed: {result}",
                error_code=4032,
            )

        data = result.get("data", {})
        if "failed_uploads" in data and data["failed_uploads"]:
            failed_files = ", ".join(
                [f["name"] for f in data["failed_uploads"]],
            )
            raise OutputError(
                f"Partial upload failed: {failed_files}",
                error_code=4033,
            )

        # Collect uploaded file IDs
        for f in data.get("uploaded_files", []):
            if file_id := f.get("file_id"):
                uploaded_files.append(file_id)

        logger.debug(f"Uploaded {len(uploaded_files)} files")
        return uploaded_files

    except Exception as e:
        raise OutputError(
            "File upload error",
            error_code=4034,
        ) from e


def secret_part_str(value: str):
    return value[:4] + "*" * 4 + value[-4:] if len(value) > 8 else "****"


def deep_mask(data: Any) -> Any:
    """
    Recursively mask sensitive fields in data structures.

    Args:
        data: Input data to process

    Returns:
        Deep copy with sensitive fields masked
    """
    if hasattr(data, "model_dump"):
        try:
            data = data.model_dump(mode="json")
        except AttributeError:
            data = data.dict()

    if isinstance(data, Dict):
        return {
            key: (
                secret_part_str(val)
                if key.lower() in LOGGER_FILTER_FIELDS
                else deep_mask(val)
            )
            for key, val in data.items()
        }
    elif isinstance(data, list):
        return [deep_mask(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(deep_mask(item) for item in data)
    elif isinstance(data, set):
        return {deep_mask(item) for item in data}
    else:
        return copy.deepcopy(data)


def set_api_key(api_key: Optional[str] = None) -> None:
    """
    Sets the DashScope API key as an environment variable.

    Args:
        api_key: The API key to set. If None, it attempts to use the
                 existing DASHSCOPE_API_KEY environment variable.

    Raises:
        ConfigurationError: If api_key is not provided and DASHSCOPE_API_KEY
                    environment variable is not set.
    """
    # 1. If api_key is provided, set it and log
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
        logger.debug(
            f"Set environ DASHSCOPE_API_KEY: "
            f"{api_key if LOG_LEVEL=='DEBUG' else deep_mask(api_key)}",
        )
        return

    # 2. If api_key is NOT provided, check if env var exists
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise ConfigurationError(
            "DashScope API key is missing. "
            "Please provide 'api_key' argument or set the "
            "'DASHSCOPE_API_KEY' environment variable.",
            error_code=4035,
        )

    # 3. If env var exists, just log that we are using it (optional)
    logger.debug("Using existing DASHSCOPE_API_KEY from environment.")


def get_filepath_classname(full_path: str) -> Tuple[str, str]:
    """
    Extract file path and class name from a path string.

    Args:
        full_path: Path in either format:
            - 'module.path.ClassName'
            - 'path/to/file.py:ClassName'

    Returns:
        Tuple (filepath, classname)

    Raises:
        InputError: For invalid formats
    """
    full_path = full_path.strip()

    if ":" in full_path:
        # Format: 'path/to/file.py:ClassName'
        parts = full_path.split(":", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise InputError(
                f"Invalid format '{full_path}'. Expected "
                f"'path/to/file.py:ClassName'",
                error_code=4041,
            )
        if ":" in parts[1]:
            raise InputError(
                f"Invalid class name format '{parts[1]}'. "
                f"Class name cannot contain colon.",
                error_code=4042,
            )

        filepath, classname = parts[0].strip(), parts[1].strip()
        if not filepath.endswith(".py"):
            filepath += ".py"
    else:
        # Format: 'module.path.ClassName'
        parts = full_path.split(".")
        if len(parts) < 2:
            raise InputError(
                f"Invalid format '{full_path}'. Expected "
                f"'module.path.ClassName' or 'path/to/file.py:ClassName'",
                error_code=4043,
            )
        classname = parts[-1]
        module_path = ".".join(parts[:-1])
        filepath = module_path.replace(".", "/") + ".py"

    return filepath.replace("\\", "/"), classname


def get_func_type_id(func_type: FunctionType):
    return str(func_type).lower() + "_id"


def _is_empty(v):
    """Return True if v is considered empty."""
    if v is None:
        return True
    if isinstance(v, str) and v == "":
        return True
    if isinstance(v, (list, dict, tuple)) and len(v) == 0:
        return True
    return False


def deep_remove_none(obj):
    """Recursively remove keys/items with empty values (None, '', [], {},
    ())."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v_cleaned = deep_remove_none(v)
            if not _is_empty(v_cleaned):  # keep non-empty items only
                cleaned[k] = v_cleaned
        return cleaned
    elif isinstance(obj, list):
        cleaned = [deep_remove_none(item) for item in obj]
        return [item for item in cleaned if not _is_empty(item)]
    else:
        return obj


def get_weights_from_file(
    filepath: str,
    classname: str = "",
) -> Dict[str, float]:
    """
    Extract reward weights from a Python file

    Args:
        filepath: Path to the Python file
        classname: Optional class name to filter by

    Returns:
        Dictionary mapping reward function names to their weights
    """
    # Check if file exists
    if not filepath or not os.path.exists(filepath):
        logger.error(f"File not found or empty filepath: {filepath}")
        return {}

    # Read file content
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}")
        return {}

    # Try AST method first
    return extract_reward_weights(source_code, classname)


def extract_reward_weights(
    source_code: str,
    classname: str,
) -> Dict[str, float]:
    """
    Static analyzer to extract reward weights from decorated functions.

    Args:
        source_code: Python source code containing decorated reward functions
        classname: Name of the class to scan

    Returns:
        Dictionary of {function_name: weight} pairs
    """

    def _parse_decorator_args(decorator) -> Dict[str, Any]:
        """Extract name and sub_weight from a decorator AST node."""
        args_dict = {}
        if isinstance(decorator, ast.Call):
            # Process args and keywords
            for i, arg in enumerate(decorator.args):
                if i == 0:
                    args_dict["name"] = _resolve_str_literal(arg)
                elif i == 1:
                    args_dict["sub_weight"] = _resolve_numeric_literal(arg)
            for kw in decorator.keywords:
                if kw.arg == "name":
                    args_dict["name"] = _resolve_str_literal(kw.value)
                elif kw.arg == "sub_weight":
                    args_dict["sub_weight"] = _resolve_numeric_literal(
                        kw.value,
                    )
        return args_dict

    def _resolve_str_literal(node) -> Optional[str]:
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Num):
            return str(node.n)
        return None

    def _resolve_numeric_literal(node) -> Optional[float]:
        if isinstance(node, ast.Num):
            val = node.n
            if isinstance(val, complex):
                return None
            return float(val)
        elif isinstance(node, ast.Constant) and isinstance(
            node.value,
            (int, float),
        ):
            return node.value
        return None

    weights = {}

    # Parse the AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return weights

    # Find the class definition
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != classname:
            continue

        for item in node.body:
            # Check for both synchronous and asynchronous functions
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for decorator in item.decorator_list:
                decorator_args = _parse_decorator_args(decorator)

                # Extract values with fallbacks
                name = decorator_args.get("name", item.name)
                weight = float(decorator_args.get("sub_weight", 1.0))

                if name:
                    weights[name] = weight

    return weights


def serialize_for_output(data: Any) -> Any:
    """
    Safely serialize various data types for output formatting.

    This function recursively processes data to ensure it can be serialized
    to formats like JSON.
    It handles:
    - Pydantic V2 models using model_dump()
    - Pydantic V1 models using dict()
    - Regular objects via their __dict__ attribute
    - Lists, tuples, and dictionaries recursively
    - Other basic types as-is

    Args:
        data: Input data to serialize (any type)

    Returns:
        Serialized data in a format suitable for output (Dict, list,
        or primitive)
    """
    # Handle Pydantic models (version detection)
    if hasattr(data, "model_dump"):  # Pydantic V2
        return data.model_dump()
    elif hasattr(data, "dict"):  # Pydantic V1
        return data.dict()

    # Handle regular objects via their attribute dictionary
    if hasattr(data, "__dict__"):
        data = data.__dict__

    # Recursively process container types
    if isinstance(data, Dict):
        return {k: serialize_for_output(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return [serialize_for_output(item) for item in data]

    # Return basic types directly
    return data
