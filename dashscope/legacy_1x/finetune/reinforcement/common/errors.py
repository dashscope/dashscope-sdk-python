# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The unchanged 1.x Agentic RL exception classes."""

from dashscope.applications.finetune.reinforcement.common._legacy_errors import (
    AgenticRLError,
    BaseConnectionError,
    BasePermissionError,
    ConfigurationError,
    DatasetsError,
    DeploymentError,
    FunctionLayerError,
    FunctionLoadError,
    IOErrorWithCode,
    InputError,
    InstanceQueryError,
    InstanceWarmupError,
    OSSConnectionError,
    OSSUploadError,
    OutputError,
    RegistrationError,
    RuntimeErrorWithCode,
    ValidationError,
    ValueErrorWithCode,
)

__all__ = [
    "AgenticRLError",
    "BaseConnectionError",
    "BasePermissionError",
    "ConfigurationError",
    "DatasetsError",
    "DeploymentError",
    "FunctionLayerError",
    "FunctionLoadError",
    "IOErrorWithCode",
    "InputError",
    "InstanceQueryError",
    "InstanceWarmupError",
    "OSSConnectionError",
    "OSSUploadError",
    "OutputError",
    "RegistrationError",
    "RuntimeErrorWithCode",
    "ValidationError",
    "ValueErrorWithCode",
]

# Preserve the historical module identity for repr/pickle/introspection even
# though the class bodies are single-sourced beside the shared core.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
