# -*- coding: utf-8 -*-
import os

import setuptools

package_root = os.path.abspath(os.path.dirname(__file__))

name = "dashscope"

description = "dashscope client sdk library"


def get_version():
    version_file = os.path.join(package_root, name, "version.py")
    version_ns = {}
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"), version_ns)
    return version_ns["__version__"]


def get_dependencies(fname="requirements.txt"):
    with open(
        fname,
        "r",
        encoding="utf-8",
    ) as f:  # pylint: disable=unspecified-encoding
        dependencies = f.readlines()
        return dependencies


url = "https://dashscope.aliyun.com/"


def readme():
    with open(os.path.join(package_root, "README.md"), encoding="utf-8") as f:
        content = f.read()
    return content


setuptools.setup(
    name=name,
    version=get_version(),
    description=description,
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Alibaba Cloud",
    author_email="dashscope@alibabacloud.com",
    license="Apache 2.0",
    url=url,
    packages=setuptools.find_packages(
        exclude=("tests"),
    ),  # pylint: disable=superfluous-parens
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    platforms="Posix; MacOS X; Windows",
    python_requires=">=3.8.0",
    install_requires=get_dependencies(),
    include_package_data=True,
    extras_require={
        "tokenizer": ["tiktoken"],
        # Interactive AI assistant (dashscope.acli); typer/rich are core deps
        "acli": [
            "prompt-toolkit>=3.0",
            "textual>=0.50",
            "PyYAML>=6.0",
            "tomli>=2.0; python_version < '3.11'",
        ],
        "acli-anthropic": ["anthropic>=0.40"],
        "acli-openai": ["openai>=1.30"],
        "acli-voice": ["sounddevice>=0.4", "numpy>=1.20"],
        "acli-camera": ["opencv-python>=4.5"],
        "acli-all": [
            "prompt-toolkit>=3.0",
            "textual>=0.50",
            "PyYAML>=6.0",
            "tomli>=2.0; python_version < '3.11'",
            "anthropic>=0.40",
            "openai>=1.30",
            "sounddevice>=0.4",
            "numpy>=1.20",
            "opencv-python>=4.5",
        ],
        # Agentic RL fine-tuning (dashscope.finetune.reinforcement)
        "rl": [
            "pydantic>=2.0",
            "tenacity",
            "PyYAML>=6.0",
            "fastapi>=0.100",
            "uvicorn>=0.20",
            "opentelemetry-sdk>=1.20",
        ],
    },
    zip_safe=False,
    entry_points={"console_scripts": ["dashscope = dashscope.cli:main"]},
)
