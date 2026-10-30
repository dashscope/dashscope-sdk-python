# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The historical Agentic RL facade over the shared application core.

Models, processors and workflow implementation remain single-sourced under
``dashscope.applications``. The 1.x path re-exports those objects but keeps the
original exception classes.
"""

from dashscope.applications.finetune import reinforcement as _core
from dashscope.legacy_1x.finetune.reinforcement.common import errors as _errors

_ERROR_NAMES = {
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
}

# Avoid resolving the lazy ``app`` attribute while the AgenticRL facade is
# importing: it imports the legacy CLI, which imports AgenticRL back.
for _name in _core.__all__:
    if _name not in _ERROR_NAMES and _name != "app":
        globals()[_name] = getattr(_core, _name)
for _name in _ERROR_NAMES:
    globals()[_name] = getattr(_errors, _name)

__all__ = list(_core.__all__)


def __getattr__(name):
    if name == "app":
        from dashscope.legacy_1x.cli.agentic_rl import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
