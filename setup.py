# -*- coding: utf-8 -*-
import os
import shutil

import setuptools
from setuptools.command.build_py import build_py as _build_py

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


class build_py(_build_py):
    """Ship the top-level ``examples/`` tree inside the wheel.

    ``examples/`` lives at the repository root (a sibling of
    ``dashscope/``) for visibility, but a wheel can only bundle files
    that sit inside a package directory. Copy the tree into
    ``<build_lib>/dashscope/examples`` so an installed wheel exposes
    ``site-packages/dashscope/examples/``.
    """

    def run(self):
        super().run()
        src = os.path.join(package_root, "examples")
        dst = os.path.join(self.build_lib, "dashscope", "examples")
        if not os.path.isdir(src):
            return
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


# Hoisted to a module-level name (like ``name``/``url`` above) so the
# static packaging audit in tests/unit/test_packaging_metadata.py keeps
# seeing only literal-evaluable setup() kwargs.
cmdclass = {"build_py": build_py}


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
        exclude=("tests", "tests.*"),
    ),  # pylint: disable=superfluous-parens
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    platforms="Posix; MacOS X; Windows",
    python_requires=">=3.9",
    install_requires=get_dependencies(),
    include_package_data=True,
    extras_require={
        "tokenizer": ["tiktoken"],
        # Interactive AI assistant (dashscope.acli); typer/rich are core deps
        "acli": [
            "prompt-toolkit>=3.0",
            "textual>=0.50,<9",
            "PyYAML>=6.0",
            "tomli>=2.0; python_version < '3.11'",
        ],
        "acli-anthropic": ["anthropic>=0.40,<2"],
        "acli-openai": ["openai>=1.30,<4"],
        "acli-voice": ["sounddevice>=0.4", "numpy>=1.20"],
        "acli-camera": ["opencv-python-headless>=4.5"],
        "acli-all": [
            "prompt-toolkit>=3.0",
            "textual>=0.50,<9",
            "PyYAML>=6.0",
            "tomli>=2.0; python_version < '3.11'",
            "anthropic>=0.40,<2",
            "openai>=1.30,<4",
            "sounddevice>=0.4",
            "numpy>=1.20",
            "opencv-python-headless>=4.5",
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
    cmdclass=cmdclass,
    entry_points={"console_scripts": ["dashscope = dashscope.cli:main"]},
)
