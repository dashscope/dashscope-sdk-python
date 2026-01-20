# -*- coding: utf-8 -*-
import os

import setuptools

package_root = os.path.abspath(os.path.dirname(__file__))

name = "dashscope"

description = "dashscope client sdk library"


def get_version():
    version_file = os.path.join(package_root, name, "version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


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
    },
    zip_safe=False,
    entry_points={"console_scripts": ["dashscope = dashscope.cli:main"]},
)
