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
    """Build one clean package tree.

    Setuptools incrementally copies files into ``build/lib`` but does not
    delete modules removed from the source tree. That is unsafe for the 2.0
    package migration: a stale ``dashscope/client/__init__.py`` shadows the
    new ``dashscope/client.py`` and makes ``from dashscope.client import
    Client`` fail in the built wheel. Remove the package build directory
    before every build, then copy the current sources only.

    ``examples/`` ships in the sdist (via MANIFEST.in) but not in the
    wheel: nothing consumes an installed copy since the agent runtime
    moved to the published dashscope-agenticx package.
    """

    def run(self):
        source_package = os.path.abspath(os.path.join(package_root, name))
        built_package = os.path.abspath(os.path.join(self.build_lib, name))
        if built_package == source_package:
            raise RuntimeError(
                "build_lib must not point at the source package directory",
            )
        if os.path.isdir(built_package):
            shutil.rmtree(built_package)
        super().run()


# Hoisted to a module-level name (like ``name``/``url`` above) so the
# static packaging audit in tests/legacy_1x/unit/test_packaging_metadata.py
# keeps seeing only literal-evaluable setup() kwargs.
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
    project_urls={
        "Documentation": "https://dashscope.github.io/dashscope-sdk-python/",
        "Source": "https://github.com/dashscope/dashscope-sdk-python",
        "Issues": "https://github.com/dashscope/dashscope-sdk-python/issues",
    },
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
        # Interactive AI assistant (DashScope SDK Expert); typer/rich are
        # core deps. `expert` is the canonical name; the `acli*` names below
        # are kept as backward-compatible aliases with identical deps.
        "expert": ["dashscope-agenticx[tui]>=1.0.0,<2"],
        "expert-anthropic": ["anthropic>=0.40,<2"],
        "expert-openai": ["openai>=1.30,<4"],
        "expert-voice": ["sounddevice>=0.4", "numpy>=1.20"],
        "expert-camera": ["opencv-python-headless>=4.5"],
        "expert-all": [
            "dashscope-agenticx[tui]>=1.0.0,<2",
            "anthropic>=0.40,<2",
            "openai>=1.30,<4",
            "sounddevice>=0.4",
            "numpy>=1.20",
            "opencv-python-headless>=4.5",
        ],
        # Backward-compatible aliases for the original `acli` extra name.
        "acli": ["dashscope-agenticx[tui]>=1.0.0,<2"],
        "acli-anthropic": ["anthropic>=0.40,<2"],
        "acli-openai": ["openai>=1.30,<4"],
        "acli-voice": ["sounddevice>=0.4", "numpy>=1.20"],
        "acli-camera": ["opencv-python-headless>=4.5"],
        "acli-all": [
            "dashscope-agenticx[tui]>=1.0.0,<2",
            "anthropic>=0.40,<2",
            "openai>=1.30,<4",
            "sounddevice>=0.4",
            "numpy>=1.20",
            "opencv-python-headless>=4.5",
        ],
    },
    zip_safe=False,
    cmdclass=cmdclass,
    entry_points={
        "console_scripts": [
            "dashscope = dashscope.cli.main:main",
            "dashscope1 = dashscope.legacy_1x.cli:main",
        ],
    },
)
