# -*- coding: utf-8 -*-
import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp
import pytest
from pydantic import BaseModel

from dashscope.finetune.reinforcement import FunctionType
from dashscope.finetune.reinforcement import InputError, ConfigurationError
from dashscope.finetune.reinforcement.common.errors import (
    RuntimeErrorWithCode,
)
from dashscope.finetune.reinforcement.common.utils import (
    async_http_request,
    generate_agentic_script,
    create_deployment_files,
    get_filepath_classname,
    deep_mask,
    set_api_key,
)


class SampleModel(BaseModel):
    api_key: str
    normal_field: str
    nested: dict


@pytest.fixture(autouse=True)
def clean_env():
    """Clean DASHSCOPE_API_KEY environment variable before and after each
    test."""
    original_key = os.environ.get("DASHSCOPE_API_KEY")
    if "DASHSCOPE_API_KEY" in os.environ:
        del os.environ["DASHSCOPE_API_KEY"]
    yield
    if original_key:
        os.environ["DASHSCOPE_API_KEY"] = original_key


class TestAgenticRLUtils:
    def test_generate_agentic_script(self):
        """Test generating a deployment script with all required components."""
        script = generate_agentic_script(
            fc_pypi_lib="dashscope",
            fc_pypi_repo="https://pypi.org/simple",
            requirements_path="requirements.txt",
            func_type="rollout",
            classpath="module.path.Processor",
        )

        # Validate core configuration
        assert 'SERVICE_TYPE="rollout"' in script
        assert 'PROCESSOR_CLASS="module.path.Processor"' in script
        assert 'SDK_PACKAGE="dashscope"' in script
        assert 'REQUIREMENTS_FILE="requirements.txt"' in script

        # Validate key functional blocks
        assert "set -euo pipefail" in script  # Error handling
        assert "install_with_retry" in script  # Retry logic
        assert "python3 -m pip install" in script  # Dependency installation
        assert 'python3 -m "${SERVER_CLASSPATH}"' in script  # Service startup

    def test_generate_script_with_empty_requirements(self):
        """Test generating a script without a requirements.txt file."""
        script = generate_agentic_script(
            fc_pypi_lib="dashscope",
            fc_pypi_repo="https://pypi.org/simple",
            requirements_path="",
            func_type="reward",
            classpath="rewards.scoring.RewardCalculator",
        )

        # Validate conditional logic for missing requirements
        assert 'if [ -f "${REQUIREMENTS_FILE}" ]; then' in script
        assert 'for pkg in "${local_packages[@]}"; do' in script
        assert "local_packages=($SDK_PACKAGE)" in script

    def test_special_characters_in_classpath(self):
        """Test handling of special characters in classpath."""
        script = generate_agentic_script(
            fc_pypi_lib="dashscope",
            fc_pypi_repo="https://pypi.org/simple",
            requirements_path="reqs.txt",
            func_type="group_reward",
            classpath="special.module:Class$With@Chars",
        )

        assert 'PROCESSOR_CLASS="special.module:Class$With@Chars"' in script

    def test_create_deployment_files(self, tmp_path):
        """Test creating deployment files with requirements."""
        # Create test environment
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()

        # Create test files
        (test_dir / "processor.py").write_text("class DemoProcessor: pass")
        (test_dir / "requirements.txt").write_text("dashscope==1.0")

        # Execute function
        create_deployment_files(
            functype=FunctionType.ROLLOUT,
            dirpath=str(test_dir),
            filepath="processor.py",
            classname="DemoProcessor",
            requirements_path="requirements.txt",
        )

        # Validate generated file (written to cwd by create_deployment_files)
        try:
            assert os.path.exists("start.sh")
            with open("start.sh", "r", encoding="utf-8") as f:
                content = f.read()
                assert 'PROCESSOR_CLASS="processor.DemoProcessor"' in content
                assert 'REQUIREMENTS_FILE="requirements.txt"' in content
        finally:
            if os.path.exists("start.sh"):
                os.remove("start.sh")

    def test_create_files_without_requirements(self, tmp_path):
        """Test creating deployment files without requirements."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        (test_dir / "module").mkdir()
        (test_dir / "module" / "processor.py").write_text(
            "class MyProcessor: pass",
        )

        create_deployment_files(
            functype=FunctionType.REWARD,
            dirpath=str(test_dir),
            filepath="module/processor.py",
            classname="MyProcessor",
            requirements_path="",
        )

        # Validate generated file (written to cwd by create_deployment_files)
        try:
            assert os.path.exists("start.sh")
            with open("start.sh", "r", encoding="utf-8") as f:
                content = f.read()
                assert (
                    'PROCESSOR_CLASS="module.processor.MyProcessor"' in content
                )
                assert "local_packages=($SDK_PACKAGE)" in content
        finally:
            if os.path.exists("start.sh"):
                os.remove("start.sh")

    def test_invalid_file_path(self, tmp_path):
        """Test handling of invalid file paths."""
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()

        with pytest.raises(Exception):
            create_deployment_files(
                functype=FunctionType.GROUP_REWARD,
                dirpath=str(test_dir),
                filepath="missing.py",
                classname="MissingProcessor",
            )

    def test_get_filepath_classname_colon_format(self):
        """Test colon-separated format parsing."""
        filepath, classname = get_filepath_classname(
            "path/to/file.py:ClassName",
        )
        assert filepath == "path/to/file.py"
        assert classname == "ClassName"

    def test_get_filepath_classname_dot_format(self):
        """Test dot-separated format parsing."""
        filepath, classname = get_filepath_classname(
            "module.submodule.ClassName",
        )
        assert filepath == "module/submodule.py"
        assert classname == "ClassName"

    def test_get_filepath_classname_mixed_format(self):
        """Test mixed format parsing."""
        filepath, classname = get_filepath_classname(
            "path.with.dots/file.py:Class.Name",
        )
        assert filepath == "path.with.dots/file.py"
        assert classname == "Class.Name"

    def test_invalid_formats(self):
        """Test handling of invalid input formats."""
        with pytest.raises(InputError):
            get_filepath_classname("singleword")

        with pytest.raises(InputError):
            get_filepath_classname("path:without:class")

        with pytest.raises(InputError):
            get_filepath_classname("file.py:")  # Missing class name

    def test_deep_mask_dict(self):
        """Test masking sensitive fields in dictionaries."""
        data = {
            "api_key": "secret_value",
            "normal_field": "visible",
            "nested": {
                "password": "123456",
                "info": "public",
            },
        }

        masked = deep_mask(data)

        assert masked["api_key"] == "secr****alue"
        assert masked["normal_field"] == "visible"
        assert masked["nested"]["password"] == "****"
        assert masked["nested"]["info"] == "public"

    def test_deep_mask_pydantic_model(self):
        """Test masking sensitive fields in Pydantic models."""
        model = SampleModel(
            api_key="model_secret",
            normal_field="model_data",
            nested={"token": "model_token", "data": "model_info"},
        )

        masked = deep_mask(model)

        # Validate sensitive fields are masked
        assert masked["api_key"] == "mode****cret"
        assert masked["normal_field"] == "model_data"
        assert masked["nested"]["token"] == "model_token"
        assert masked["nested"]["data"] == "model_info"

    def test_deep_mask_complex_structure(self):
        """Test masking in complex nested structures."""
        data = [
            {"key": "value1", "password": "s1"},
            {"key": "value2", "api_token": "s2"},
            (
                {"key": "value3", "trigger_token": "s3"},
                {"key": "value4", "instance_token": "s4"},
            ),
        ]

        masked = deep_mask(data)

        assert masked[0]["password"] == "****"
        assert masked[1]["api_token"] == "****"
        assert masked[2][0]["trigger_token"] == "****"
        assert masked[2][1]["instance_token"] == "****"

    def test_set_api_key_with_argument(self):
        """Test setting API key via function argument."""
        with patch("os.environ", {}):
            set_api_key("test_key_123")
            assert os.environ["DASHSCOPE_API_KEY"] == "test_key_123"

    def test_set_api_key_with_env_var(self):
        """Test using existing environment variable."""
        with patch("os.environ", {"DASHSCOPE_API_KEY": "env_key_456"}):
            set_api_key(None)
            assert os.environ["DASHSCOPE_API_KEY"] == "env_key_456"

    def test_set_api_key_missing(self):
        """Test handling of missing API key."""
        with patch("os.environ", {}):
            with pytest.raises(ConfigurationError):
                set_api_key(None)


def _mock_session(side_effect):
    """Create a mock aiohttp.ClientSession that raises side_effect
    on any HTTP method (get/post)."""
    mock_resp_ctx = AsyncMock()
    mock_resp_ctx.__aenter__ = AsyncMock(side_effect=side_effect)
    mock_resp_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp_ctx)
    mock_session.post = MagicMock(return_value=mock_resp_ctx)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


class TestAsyncHttpRequestRootCause:
    """Verify async_http_request raises exceptions (not return dicts)
    so that root cause is preserved in __cause__ chain."""

    @pytest.mark.asyncio
    async def test_client_error_raises_with_root_cause(self):
        """ClientConnectionError should be preserved as __cause__."""
        original = aiohttp.ClientConnectionError(
            "Cannot write to transport",
        )

        with patch(
            "dashscope.finetune.reinforcement.common.utils.aiohttp"
            ".ClientSession",
            return_value=_mock_session(original),
        ):
            with pytest.raises(RuntimeErrorWithCode) as exc_info:
                await async_http_request(
                    method="POST",
                    url="https://example.com/api",
                    data={"key": "value"},
                    timeout=5,
                    retry_times=1,
                )

            assert exc_info.value.error_code == "sdk.RuntimeErrorWithCode"
            root = exc_info.value.__cause__
            assert isinstance(root, aiohttp.ClientConnectionError)
            assert "Cannot write to transport" in str(root)

    @pytest.mark.asyncio
    async def test_timeout_raises_with_root_cause(self):
        """TimeoutError should be preserved as __cause__."""
        with patch(
            "dashscope.finetune.reinforcement.common.utils.aiohttp"
            ".ClientSession",
            return_value=_mock_session(asyncio.TimeoutError()),
        ):
            with pytest.raises(RuntimeErrorWithCode) as exc_info:
                await async_http_request(
                    method="POST",
                    url="https://example.com/api",
                    data={"key": "value"},
                    timeout=5,
                    retry_times=1,
                )

            assert exc_info.value.error_code == "sdk.RuntimeErrorWithCode"
            assert isinstance(
                exc_info.value.__cause__,
                asyncio.TimeoutError,
            )

    @pytest.mark.asyncio
    async def test_root_cause_walkable(self):
        """Root cause should be reachable by walking __cause__ chain."""
        original = aiohttp.ClientConnectionError("connection reset")

        with patch(
            "dashscope.finetune.reinforcement.common.utils.aiohttp"
            ".ClientSession",
            return_value=_mock_session(original),
        ):
            with pytest.raises(RuntimeErrorWithCode) as exc_info:
                await async_http_request(
                    method="GET",
                    url="https://example.com/api",
                    timeout=5,
                    retry_times=1,
                )

            root = exc_info.value
            while root.__cause__:
                root = root.__cause__
            assert isinstance(root, aiohttp.ClientConnectionError)
            assert "connection reset" in str(root)
