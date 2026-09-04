# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import dashscope
from dashscope.cli import agentic_rl
from dashscope.cli import (
    app as cli_app,
    main as cli_main,
)
from dashscope.common.error import AuthenticationError


def strip_ansi_codes(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


# pylint: disable=too-many-public-methods
class TestCliMain:
    def test_main_prints_authentication_error_without_traceback(
        self,
        monkeypatch,
        capsys,
    ):
        def mock_list(**kwargs):
            raise AuthenticationError("No api key provided.")

        monkeypatch.setattr(
            "dashscope.cli.models.dashscope.Models.list",
            mock_list,
        )
        monkeypatch.setattr(sys, "argv", ["dashscope", "models", "list"])

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 1
        assert "No api key provided." in combined_output
        assert "Traceback" not in combined_output

    def test_top_level_help_shows_global_api_key(self):
        result = CliRunner().invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        clean_output = strip_ansi_codes(result.output)
        assert "--api-key" in clean_output

    def test_python_module_help_suppresses_urllib3_warning(self):
        result = subprocess.run(
            [sys.executable, "-m", "dashscope.cli", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        combined_output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "NotOpenSSLWarning" not in combined_output

    def test_rl_test_functions_invalid_input_without_traceback(self):
        result = CliRunner().invoke(
            agentic_rl.app,
            [
                "test-functions",
                "ins-1234",
                "--type",
                "ROLLOUT",
                "--input",
                "not-json",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid input" in result.output
        assert "Traceback" not in result.output

    def test_rl_register_functions_hyphen_alias_help(self):
        result = CliRunner().invoke(
            agentic_rl.app,
            ["register-functions", "--help"],
        )

        assert result.exit_code == 0
        clean_output = strip_ansi_codes(result.output)
        assert "rollout-classpaths" in clean_output

    def test_rl_test_functions_hyphen_alias_help(self):
        result = CliRunner().invoke(
            agentic_rl.app,
            ["test-functions", "--help"],
        )

        assert result.exit_code == 0
        clean_output = strip_ansi_codes(result.output)
        assert "--input" in clean_output

    def test_rl_upload_data_hyphen_alias_help(self):
        result = CliRunner().invoke(
            agentic_rl.app,
            ["upload-data", "--help"],
        )

        assert result.exit_code == 0
        clean_output = strip_ansi_codes(result.output)
        assert "training-files" in clean_output

    def test_missing_global_api_key_value_does_not_consume_command(
        self,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "--api-key", "models", "list"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 2
        assert "requires an argument" in combined_output
        assert "No such command 'list'" not in combined_output

    def test_missing_global_api_key_value_does_not_consume_option(
        self,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(sys, "argv", ["dashscope", "--api-key", "--help"])

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 2
        assert "requires an argument" in combined_output
        assert "Usage:" not in combined_output

    def test_empty_global_api_key_value_exits_before_request(
        self,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(dashscope, "api_key", "existing-key")
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "--api-key=", "models", "list"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 2
        assert dashscope.api_key == "existing-key"
        assert "requires an argument" in combined_output
        assert "InvalidApiKey" not in combined_output

    def test_global_api_key_after_command_is_supported(self, monkeypatch):
        captured_request = {}

        def mock_list(**kwargs):
            captured_request["api_key"] = dashscope.api_key
            captured_request.update(kwargs)
            return SimpleNamespace(status_code=200, output={"models": []})

        monkeypatch.setattr(
            "dashscope.cli.models.dashscope.Models.list",
            mock_list,
        )
        monkeypatch.setattr(dashscope, "api_key", None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "models", "--api-key", "command-key", "list"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        assert exception_info.value.code == 0
        assert captured_request == {
            "api_key": "command-key",
            "page": 1,
            "page_size": 10,
        }

    def test_global_api_key_equals_after_command_is_supported(
        self,
        monkeypatch,
    ):
        captured_request = {}

        def mock_list(**kwargs):
            captured_request["api_key"] = dashscope.api_key
            captured_request.update(kwargs)
            return SimpleNamespace(status_code=200, output={"models": []})

        monkeypatch.setattr(
            "dashscope.cli.models.dashscope.Models.list",
            mock_list,
        )
        monkeypatch.setattr(dashscope, "api_key", None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "models", "--api-key=command-key", "list"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        assert exception_info.value.code == 0
        assert captured_request == {
            "api_key": "command-key",
            "page": 1,
            "page_size": 10,
        }

    def test_agentic_rl_hidden_alias_help(self):
        result = CliRunner().invoke(cli_app, ["agentic-rl", "--help"])

        assert result.exit_code == 0
        assert "register_functions" in result.output

    def test_subcommand_api_key_option_is_not_consumed_by_global_parser(
        self,
        monkeypatch,
    ):
        captured_request = {}

        def mock_upload(model, file_path, api_key, base_address=None):
            captured_request["model"] = model
            captured_request["file_path"] = file_path
            captured_request["api_key"] = api_key
            captured_request["base_address"] = base_address
            return "oss://uploaded", None

        monkeypatch.setattr(
            "dashscope.cli.oss.os.path.exists",
            lambda path: True,
        )
        monkeypatch.setattr("dashscope.cli.oss.OssUtils.upload", mock_upload)
        monkeypatch.setattr(dashscope, "api_key", "global-key")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dashscope",
                "oss",
                "upload",
                "--api-key",
                "local-key",
                "--file",
                "~/file.txt",
                "--model",
                "qwen-vl",
            ],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        assert exception_info.value.code == 0
        assert captured_request == {
            "model": "qwen-vl",
            "file_path": os.path.expanduser("~/file.txt"),
            "api_key": "local-key",
            "base_address": None,
        }

    def test_subcommand_api_key_missing_value_is_handled_by_subcommand_parser(
        self,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dashscope",
                "oss",
                "upload",
                "--api-key",
                "--file",
                "~/file.txt",
                "--model",
                "qwen-vl",
            ],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 2
        assert "requires an argument" in combined_output

    def test_legacy_api_key_missing_value_exits_before_request(
        self,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dashscope",
                "generation.call",
                "--api_key",
                "-m",
                "qwen-turbo",
                "-p",
                "hi",
            ],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 2
        assert "requires an argument" in combined_output

    def test_legacy_api_key_option_is_extracted_after_translation(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(dashscope, "api_key", None)
        captured_request = {}

        def mock_call(model, prompt, stream=False):
            captured_request["api_key"] = dashscope.api_key
            captured_request["model"] = model
            captured_request["prompt"] = prompt
            captured_request["stream"] = stream
            return SimpleNamespace(
                status_code=200,
                output={"text": "ok"},
                usage={"input_tokens": 1},
            )

        monkeypatch.setattr(
            "dashscope.cli.generation.Generation.call",
            mock_call,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dashscope",
                "generation.call",
                "--api_key",
                "legacy-key",
                "-m",
                "qwen-turbo",
                "-p",
                "hi",
            ],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        assert exception_info.value.code == 0
        assert captured_request == {
            "api_key": "legacy-key",
            "model": "qwen-turbo",
            "prompt": "hi",
            "stream": False,
        }

    def test_legacy_list_page_size_maps_to_size_option(self, monkeypatch):
        captured_requests = {}

        def mock_files_list(**kwargs):
            captured_requests["files"] = kwargs
            return SimpleNamespace(status_code=200, output={})

        def mock_fine_tunes_list(**kwargs):
            captured_requests["fine_tunes"] = kwargs
            from dashscope.finetune.customize_types import (
                FineTuneList,
                FineTuneListOutput,
            )

            result = FineTuneList(status_code=200)
            result.output = FineTuneListOutput(jobs=[])
            return result

        def mock_deployments_list(**kwargs):
            captured_requests["deployments"] = kwargs
            from dashscope.finetune.customize_types import (
                DeploymentList,
                DeploymentListOutput,
            )

            result = DeploymentList(status_code=200)
            result.output = DeploymentListOutput(deployments=[])
            return result

        monkeypatch.setattr(
            "dashscope.cli.files.dashscope.Files.list",
            mock_files_list,
        )
        monkeypatch.setattr(
            "dashscope.cli.fine_tunes.dashscope.FineTunes.list",
            mock_fine_tunes_list,
        )
        monkeypatch.setattr(
            "dashscope.cli.deployments.dashscope.Deployments.list",
            mock_deployments_list,
        )

        cases = [
            ("files.list", "files"),
            ("fine_tunes.list", "fine_tunes"),
            ("deployments.list", "deployments"),
        ]
        for legacy_command, request_key in cases:
            monkeypatch.setattr(
                sys,
                "argv",
                ["dashscope", legacy_command, "--page_size", "20"],
            )

            with pytest.raises(SystemExit) as exception_info:
                cli_main()

            assert exception_info.value.code == 0
            assert captured_requests[request_key]["page_size"] == 20

    def test_legacy_list_page_size_equals_maps_to_size_option(
        self,
        monkeypatch,
    ):
        captured_request = {}

        def mock_files_list(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(status_code=200, output={})

        monkeypatch.setattr(
            "dashscope.cli.files.dashscope.Files.list",
            mock_files_list,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "files.list", "--page_size=30"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        assert exception_info.value.code == 0
        assert captured_request["page_size"] == 30

    def test_files_upload_missing_file_exits_without_traceback(
        self,
        monkeypatch,
        capsys,
    ):
        monkeypatch.setattr(
            "dashscope.cli.files.os.path.exists",
            lambda path: False,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "files", "upload", "--file", "~/missing.jsonl"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 1
        assert "does not exist" in combined_output
        assert "Traceback" not in combined_output

    def test_files_upload_sdk_exception_exits_without_traceback(
        self,
        monkeypatch,
        capsys,
    ):
        def mock_upload(**kwargs):
            raise RuntimeError("upload failed")

        monkeypatch.setattr(
            "dashscope.cli.files.os.path.exists",
            lambda path: True,
        )
        monkeypatch.setattr(
            "dashscope.cli.files.dashscope.Files.upload",
            mock_upload,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "files", "upload", "--file", "~/data.jsonl"],
        )

        with pytest.raises(SystemExit) as exception_info:
            cli_main()

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.code == 1
        assert "upload failed" in combined_output
        assert "Traceback" not in combined_output

    def test_agentic_rl_invalid_output_format_exits(self, capsys):
        with pytest.raises(Exception) as exception_info:
            agentic_rl.format_output({"job_id": "job-1"}, "xml")

        captured_output = capsys.readouterr()
        combined_output = captured_output.out + captured_output.err

        assert exception_info.value.exit_code == 2
        assert "Invalid output format" in combined_output

    def test_agentic_rl_invalid_output_format_exits_before_business_validation(
        self,
    ):
        result = CliRunner().invoke(
            cli_app,
            ["rl", "register_functions", "--output-format", "xml"],
        )

        assert result.exit_code == 2
        assert "Invalid output format" in result.output
        assert "At least one" not in result.output

    def test_agentic_rl_invalid_function_type_exits_before_sdk_call(self):
        result = CliRunner().invoke(
            cli_app,
            ["rl", "test_functions", "iid", "--type", "bad", "--input", "{}"],
        )

        assert result.exit_code == 2
        assert "Invalid function type" in result.output
        assert "Function test failed" not in result.output
        assert "Traceback" not in result.output

    def test_agentic_rl_upload_data_missing_file_exits_before_sdk_call(self):
        result = CliRunner().invoke(
            cli_app,
            ["rl", "upload_data", "--training-files", "~/missing.jsonl"],
        )

        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert "Dataset upload failed" not in result.output
        assert "Traceback" not in result.output

    def test_generation_stream_exception_is_printed_without_traceback(
        self,
        monkeypatch,
    ):
        class FailingStream:
            def __iter__(self):
                raise RuntimeError("stream failed")

        def mock_call(*_args, **_kwargs):
            return FailingStream()

        monkeypatch.setattr(
            "dashscope.cli.generation.Generation.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            [
                "generation",
                "create",
                "--model",
                "qwen",
                "--prompt",
                "hi",
                "--stream",
            ],
        )

        assert result.exit_code == 1
        assert "Generation request failed: stream failed" in result.output
        assert "Traceback" not in result.output

    def test_models_sdk_exception_is_printed_without_traceback(
        self,
        monkeypatch,
    ):
        def mock_list(**kwargs):
            raise RuntimeError("models failed")

        monkeypatch.setattr(
            "dashscope.cli.models.dashscope.Models.list",
            mock_list,
        )

        result = CliRunner().invoke(cli_app, ["models", "list"])

        assert result.exit_code == 1
        assert "List models failed: models failed" in result.output
        assert "Traceback" not in result.output

    def test_deployments_sdk_exception_is_printed_without_traceback(
        self,
        monkeypatch,
    ):
        def mock_call(**kwargs):
            raise RuntimeError("deployment failed")

        monkeypatch.setattr(
            "dashscope.cli.deployments.dashscope.Deployments.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            ["deployments", "create", "--model", "qwen", "--plan", "ptu"],
        )

        assert result.exit_code == 1
        assert "Create deployment failed: deployment failed" in result.output
        assert "Traceback" not in result.output

    def test_application_response_without_usage_is_supported(
        self,
        monkeypatch,
    ):
        def mock_call(**_kwargs):
            return SimpleNamespace(status_code=200, output={"text": "ok"})

        monkeypatch.setattr(
            "dashscope.cli.application.dashscope.Application.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            ["application", "create", "--app-id", "app", "--prompt", "hi"],
        )

        assert result.exit_code == 0
        assert "ok" in result.output
        assert "Traceback" not in result.output

    def test_generation_response_without_usage_is_supported(self, monkeypatch):
        def mock_call(*_args, **_kwargs):
            return SimpleNamespace(status_code=200, output={"text": "ok"})

        monkeypatch.setattr(
            "dashscope.cli.generation.Generation.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            ["generation", "create", "--model", "qwen", "--prompt", "hi"],
        )

        assert result.exit_code == 0
        assert "ok" in result.output
        assert "Traceback" not in result.output

    def test_generation_stream_response_without_usage_is_supported(
        self,
        monkeypatch,
    ):
        def mock_call(*_args, **_kwargs):
            return iter(
                [SimpleNamespace(status_code=200, output={"text": "ok"})],
            )

        monkeypatch.setattr(
            "dashscope.cli.generation.Generation.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            [
                "generation",
                "create",
                "--model",
                "qwen",
                "--prompt",
                "hi",
                "--stream",
            ],
        )

        assert result.exit_code == 0
        assert "ok" in result.output
        assert "Traceback" not in result.output

    def test_image_generation_sdk_exception_is_printed_without_traceback(
        self,
        monkeypatch,
    ):
        def mock_call(**kwargs):
            raise RuntimeError("image failed")

        monkeypatch.setattr(
            "dashscope.cli.image_generation.ImageGeneration.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            [
                "image-generation",
                "create",
                "--model",
                "qwen-image",
                "--text",
                "cat",
            ],
        )

        assert result.exit_code == 1
        assert "Image generation request failed: image failed" in result.output
        assert "Traceback" not in result.output

    def test_multimodal_conversation_response_without_usage_is_supported(
        self,
        monkeypatch,
    ):
        def mock_call(**_kwargs):
            return SimpleNamespace(status_code=200, output={"text": "ok"})

        monkeypatch.setattr(
            "dashscope.cli.multimodal_conversation.dashscope."
            "MultiModalConversation.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            [
                "multimodal-conversation",
                "create",
                "--model",
                "qwen-vl",
                "--text",
                "hi",
            ],
        )

        assert result.exit_code == 0
        assert "ok" in result.output
        assert "Traceback" not in result.output

    def test_multimodal_embedding_response_without_usage_is_supported(
        self,
        monkeypatch,
    ):
        def mock_call(**_kwargs):
            return SimpleNamespace(status_code=200, output={"embeddings": []})

        monkeypatch.setattr(
            "dashscope.cli.multimodal_embedding.dashscope."
            "MultiModalEmbedding.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            [
                "multimodal-embedding",
                "create",
                "--model",
                "mm-embedding",
                "--text",
                "hi",
            ],
        )

        assert result.exit_code == 0
        assert "embeddings" in result.output
        assert "Traceback" not in result.output

    def test_multimodal_embedding_sdk_exception_is_printed_without_traceback(
        self,
        monkeypatch,
    ):
        def mock_call(**_kwargs):
            raise RuntimeError("embedding failed")

        monkeypatch.setattr(
            "dashscope.cli.multimodal_embedding.dashscope."
            "MultiModalEmbedding.call",
            mock_call,
        )

        result = CliRunner().invoke(
            cli_app,
            [
                "multimodal-embedding",
                "create",
                "--model",
                "mm-embedding",
                "--text",
                "hi",
            ],
        )

        assert result.exit_code == 1
        assert (
            "Multimodal embedding request failed: embedding failed"
            in result.output
        )
        assert "Traceback" not in result.output

    def test_agentic_rl_run_missing_config_exits_before_workflow(self):
        result = CliRunner().invoke(
            cli_app,
            ["rl", "run", "--config", "~/missing-config.yaml"],
        )

        assert result.exit_code == 1
        assert "--config file" in result.output
        assert "does not exist" in result.output
        assert "Workflow execution failed" not in result.output
        assert "Traceback" not in result.output
