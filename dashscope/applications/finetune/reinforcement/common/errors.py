# -*- coding: utf-8 -*-
# pylint: disable=too-many-ancestors
"""Native v2 errors for AgenticRL, with legacy numeric-code compatibility."""

from datetime import datetime
from typing import Any, Dict, Optional

from dashscope.errors import _CatalogSdkError, sdk_error

from ._legacy_errors import (
    AgenticRLError as _LegacyAgenticRLError,
    BaseConnectionError as _LegacyBaseConnectionError,
    BasePermissionError as _LegacyBasePermissionError,
    ConfigurationError as _LegacyConfigurationError,
    DatasetsError as _LegacyDatasetsError,
    DeploymentError as _LegacyDeploymentError,
    FunctionLayerError as _LegacyFunctionLayerError,
    FunctionLoadError as _LegacyFunctionLoadError,
    IOErrorWithCode as _LegacyIOErrorWithCode,
    InputError as _LegacyInputError,
    InstanceQueryError as _LegacyInstanceQueryError,
    InstanceWarmupError as _LegacyInstanceWarmupError,
    OSSConnectionError as _LegacyOSSConnectionError,
    OSSUploadError as _LegacyOSSUploadError,
    OutputError as _LegacyOutputError,
    RegistrationError as _LegacyRegistrationError,
    RuntimeErrorWithCode as _LegacyRuntimeErrorWithCode,
    ValidationError as _LegacyValidationError,
    ValueErrorWithCode as _LegacyValueErrorWithCode,
)

_LEGACY_ERROR_CODES = {
    "sdk.agentic_rl.Error": 1000,
    "sdk.agentic_rl.IOErrorWithCode": 1800,
    "sdk.agentic_rl.RuntimeErrorWithCode": 0,
    "sdk.agentic_rl.ValueErrorWithCode": 0,
    "sdk.agentic_rl.InputError": 1100,
    "sdk.agentic_rl.OutputError": 1200,
    "sdk.agentic_rl.BaseConnectionError": 1300,
    "sdk.agentic_rl.OSSConnectionError": 1310,
    "sdk.agentic_rl.OSSUploadError": 1320,
    "sdk.agentic_rl.DeploymentError": 1400,
    "sdk.agentic_rl.RegistrationError": 1410,
    "sdk.agentic_rl.DatasetsError": 1460,
    "sdk.agentic_rl.FunctionLoadError": 1420,
    "sdk.agentic_rl.FunctionLayerError": 1450,
    "sdk.agentic_rl.InstanceWarmupError": 1430,
    "sdk.agentic_rl.InstanceQueryError": 1440,
    "sdk.agentic_rl.ValidationError": 1500,
    "sdk.agentic_rl.ConfigurationError": 1510,
    "sdk.agentic_rl.BasePermissionError": 1700,
}


class _RootCauseMixin:
    @property
    def root_cause(self) -> Exception:
        root: Exception = self  # type: ignore[assignment]
        seen = {id(root)}
        while root.__cause__ and id(root.__cause__) not in seen:
            root = root.__cause__
            seen.add(id(root))
        return root

    def _format_cause(self) -> str:
        if self.__cause__ is not None:
            root = self.root_cause
            if self is not root:
                cause_msg = str(root).split("\n", maxsplit=1)[0][:100].strip()
                if cause_msg:
                    return f" (caused by: {type(root).__name__}: {cause_msg})"
                return f" (caused by: {type(root).__name__})"
        return ""


class AgenticRLError(
    _RootCauseMixin,
    _CatalogSdkError,
    _LegacyAgenticRLError,
):
    """Base class for AgenticRL failures in the v2 error family."""

    _SDK_ERROR_NAME = "sdk.agentic_rl.Error"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._legacy_error_code = error_code
        self.timestamp = datetime.now().isoformat()
        super().__init__(message, **kwargs)
        # Legacy compatibility bases run inside the cooperative MRO and set
        # their own defaults. Restore the v2/catalog selection requested by
        # this constructor after that initialization completes.
        self._legacy_error_code = error_code

    @classmethod
    def from_catalog(
        cls,
        name: str,
        *,
        error_code: int,
        variables: Optional[Dict[str, str]] = None,
    ) -> "AgenticRLError":
        error = sdk_error(name, variables)
        return cls(
            error.message,
            error_code=error_code,
            code=error.code,
            solution=error.solution,
            allow_retry=error.allow_retry,
        )

    @property
    def error_code(self) -> int:
        if self._legacy_error_code is not None:
            return self._legacy_error_code
        return _LEGACY_ERROR_CODES.get(self.code or "", 1000)

    @error_code.setter
    def error_code(self, value: int) -> None:
        self._legacy_error_code = value

    def __str__(self) -> str:
        return f"{super().__str__()}{self._format_cause()}"


class IOErrorWithCode(AgenticRLError, _LegacyIOErrorWithCode):
    _SDK_ERROR_NAME = "sdk.agentic_rl.IOErrorWithCode"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"I/O error: {message}", error_code, **kwargs)
        self.path = path
        self.operation = operation


class RuntimeErrorWithCode(  # type: ignore[misc]
    AgenticRLError,
    _LegacyRuntimeErrorWithCode,
):
    _SDK_ERROR_NAME = "sdk.agentic_rl.RuntimeErrorWithCode"


class ValueErrorWithCode(  # type: ignore[misc]
    AgenticRLError,
    _LegacyValueErrorWithCode,
):
    _SDK_ERROR_NAME = "sdk.agentic_rl.ValueErrorWithCode"


class InputError(AgenticRLError, _LegacyInputError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.InputError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        field: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code, **kwargs)
        self.field = field


class OutputError(AgenticRLError, _LegacyOutputError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.OutputError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        response: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code, **kwargs)
        self.response = response


class BaseConnectionError(AgenticRLError, _LegacyBaseConnectionError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.BaseConnectionError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code, **kwargs)
        self.endpoint = endpoint


class OSSConnectionError(BaseConnectionError, _LegacyOSSConnectionError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.OSSConnectionError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"OSS connection failed: {message}",
            error_code,
            endpoint,
            **kwargs,
        )


class OSSUploadError(BaseConnectionError, _LegacyOSSUploadError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.OSSUploadError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        bucket: Optional[str] = None,
        object_key: Optional[str] = None,
        file_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"OSS upload failed: {message}",
            error_code,
            endpoint,
            **kwargs,
        )
        self.bucket = bucket
        self.object_key = object_key
        self.file_size = file_size


class DeploymentError(AgenticRLError, _LegacyDeploymentError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.DeploymentError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code, **kwargs)
        self.resource_id = resource_id


class RegistrationError(DeploymentError, _LegacyRegistrationError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.RegistrationError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Registration failed: {message}",
            error_code,
            resource_id,
            **kwargs,
        )


class DatasetsError(DeploymentError, _LegacyDatasetsError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.DatasetsError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Datasets failed: {message}", error_code, **kwargs)


class FunctionLoadError(DeploymentError, _LegacyFunctionLoadError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.FunctionLoadError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        entity_id: Optional[str] = None,
        error_log: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Function load failed: {message}",
            error_code,
            entity_id,
            **kwargs,
        )
        self.entity_id = entity_id
        self.error_log = error_log


class FunctionLayerError(DeploymentError, _LegacyFunctionLayerError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.FunctionLayerError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        layer_name: Optional[str] = None,
        error_log: Optional[str] = None,
        layer_code: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Raised for create, status-query and build-timeout failures alike, so
        # the prefix names the subsystem rather than the create stage. The
        # archived 1.x base keeps its original wording.
        super().__init__(
            f"Function layer failed: {message}",
            error_code,
            layer_code,
            **kwargs,
        )
        self.layer_name = layer_name
        self.layer_code = layer_code
        self.error_log = error_log


class InstanceWarmupError(DeploymentError, _LegacyInstanceWarmupError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.InstanceWarmupError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        instance_url: Optional[str] = None,
        timeout: float = 0.0,
        retry_after: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Instance warmup failed: {message}",
            error_code,
            **kwargs,
        )
        self.instance_url = instance_url
        self.timeout = timeout
        self.retry_after = retry_after


class InstanceQueryError(DeploymentError, _LegacyInstanceQueryError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.InstanceQueryError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        instance_id: Optional[str] = None,
        query_attempts: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Instance query failed: {message}",
            error_code,
            instance_id,
            **kwargs,
        )
        self.instance_id = instance_id
        self.query_attempts = query_attempts


class ValidationError(AgenticRLError, _LegacyValidationError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.ValidationError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        invalid_data: Optional[Dict] = None,
        validation_rules: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Validation failed: {message}", error_code, **kwargs)
        self.invalid_data = invalid_data
        self.validation_rules = validation_rules


class ConfigurationError(ValidationError, _LegacyConfigurationError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.ConfigurationError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code=error_code, **kwargs)
        self.config_path = config_path


class BasePermissionError(AgenticRLError, _LegacyBasePermissionError):
    _SDK_ERROR_NAME = "sdk.agentic_rl.BasePermissionError"

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        operation: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(f"Permission denied: {message}", error_code, **kwargs)
        self.operation = operation
        self.resource = resource
