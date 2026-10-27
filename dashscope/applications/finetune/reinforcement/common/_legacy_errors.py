# -*- coding: utf-8 -*-
"""
Custom exception hierarchy for the AgenticRL system
"""

from datetime import datetime
from typing import Optional, Dict


class _RootCauseMixin:
    """Mixin that provides root-cause traversal and formatting for exceptions
    that carry an error code."""

    @property
    def root_cause(self) -> "Exception":
        root: "Exception" = self  # type: ignore[assignment]
        seen = {id(root)}
        while root.__cause__ and id(root.__cause__) not in seen:
            root = root.__cause__
            seen.add(id(root))
        return root

    def _format_cause(self) -> str:
        """Format root cause information if available."""
        if self.__cause__ is not None:
            root = self.root_cause
            if self is not root:
                cause_msg = str(root).split("\n", maxsplit=1)[0][:100].strip()
                if cause_msg:
                    return f" (caused by: {type(root).__name__}: {cause_msg})"
                return f" (caused by: {type(root).__name__})"
        return ""


class AgenticRLError(_RootCauseMixin, Exception):
    """Base class for all Agentic RL exceptions."""

    def __init__(self, message: str, error_code: int = 1000):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()
        self.message = message

    def __str__(self):
        base = f"[{self.error_code}] {self.message} (at {self.timestamp})"
        return f"{base}{self._format_cause()}"


class IOErrorWithCode(AgenticRLError):
    """Raised for general I/O operation failures"""

    def __init__(
        self,
        message: str,
        error_code: int = 1800,
        path: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(f"I/O error: {message}", error_code)
        self.path = path
        self.operation = operation


class RuntimeErrorWithCode(_RootCauseMixin, RuntimeError):
    """Enhanced RuntimeError that supports error codes for better error
    categorization."""

    def __init__(self, message: str, error_code: int = 0):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def __str__(self):
        return f"[{self.error_code}] {self.message}{self._format_cause()}"


class ValueErrorWithCode(_RootCauseMixin, ValueError):
    """Enhanced ValueError that supports error codes for better error
    categorization."""

    def __init__(self, message: str, error_code: int = 0):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def __str__(self):
        return f"[{self.error_code}] {self.message}{self._format_cause()}"


class InputError(AgenticRLError):
    """Raised when invalid input data is detected during validation"""

    def __init__(
        self,
        message: str,
        error_code: int = 1100,
        field: Optional[str] = None,
    ):
        super().__init__(message, error_code)
        self.field = field


class OutputError(AgenticRLError):
    """Raised when service response fails output validation checks"""

    def __init__(
        self,
        message: str,
        error_code: int = 1200,
        response: Optional[Dict] = None,
    ):
        super().__init__(message, error_code)
        self.response = response


class BaseConnectionError(AgenticRLError):
    """Base class for connection-related errors"""

    def __init__(
        self,
        message: str,
        error_code: int = 1300,
        endpoint: Optional[str] = None,
    ):
        super().__init__(message, error_code)
        self.endpoint = endpoint


class OSSConnectionError(BaseConnectionError):
    """Raised when connecting to OSS storage service fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1310,
        endpoint: str = None,
    ):
        super().__init__(
            f"OSS connection failed: {message}",
            error_code,
            endpoint,
        )


class OSSUploadError(BaseConnectionError):
    """Raised when file upload operation to OSS fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1320,
        endpoint: str = None,
        bucket: Optional[str] = None,
        object_key: Optional[str] = None,
        file_size: Optional[int] = None,
    ):
        super().__init__(f"OSS upload failed: {message}", error_code, endpoint)
        self.bucket = bucket
        self.object_key = object_key
        self.file_size = file_size


class DeploymentError(AgenticRLError):
    """Base class for deployment-related errors"""

    def __init__(
        self,
        message: str,
        error_code: int = 1400,
        resource_id: Optional[str] = None,
    ):
        super().__init__(message, error_code)
        self.resource_id = resource_id


class RegistrationError(DeploymentError):
    """Raised when function registration fails in the deployment system"""

    def __init__(
        self,
        message: str,
        error_code: int = 1410,
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            f"Registration failed: {message}",
            error_code,
            resource_id,
        )


class DatasetsError(DeploymentError):
    """Raised when update datasets fails in the deployment system"""

    def __init__(self, message: str, error_code: int = 1460):
        super().__init__(f"Datasets failed: {message}", error_code)


class FunctionLoadError(DeploymentError):
    """Raised when loading a registered function into runtime fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1420,
        entity_id: str = None,
        error_log: Optional[str] = None,
    ):
        super().__init__(
            f"Function load failed: {message}",
            error_code,
            entity_id,
        )
        self.entity_id = entity_id
        self.error_log = error_log


class FunctionLayerError(DeploymentError):
    """Raised when creating a layer of function fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1450,
        layer_name: str = None,
        error_log: Optional[str] = None,
    ):
        super().__init__(
            f"Function layer create failed: {message}",
            error_code,
            layer_name,
        )
        self.layer_name = layer_name
        self.error_log = error_log


class InstanceWarmupError(DeploymentError):
    """Raised when function instance health check fails after deployment"""

    def __init__(
        self,
        message: str,
        error_code: int = 1430,
        instance_url: str = None,
        timeout: float = 0.0,
        retry_after: Optional[float] = None,
    ):
        super().__init__(f"Instance warmup failed: {message}", error_code)
        self.instance_url = instance_url
        self.timeout = timeout
        self.retry_after = retry_after


class InstanceQueryError(DeploymentError):
    """Raised when querying function instance status fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1440,
        instance_id: str = None,
        query_attempts: int = 1,
    ):
        super().__init__(
            f"Instance query failed: {message}",
            error_code,
            instance_id,
        )
        self.instance_id = instance_id
        self.query_attempts = query_attempts


class ValidationError(AgenticRLError):
    """Base class for data validation failures"""

    def __init__(
        self,
        message: str,
        error_code: int = 1500,
        invalid_data: Optional[Dict] = None,
        validation_rules: Optional[Dict] = None,
    ):
        super().__init__(f"Validation failed: {message}", error_code)
        self.invalid_data = invalid_data
        self.validation_rules = validation_rules


class ConfigurationError(ValidationError):
    """Raised when invalid system configuration is detected"""

    def __init__(
        self,
        message: str,
        error_code: int = 1510,
        config_path: Optional[str] = None,
    ):
        super().__init__(message, error_code=error_code)
        self.config_path = config_path


class BasePermissionError(AgenticRLError):
    """Raised when an operation lacks required permissions"""

    def __init__(
        self,
        message: str,
        error_code: int = 1700,
        operation: str = None,
        resource: str = None,
    ):
        super().__init__(f"Permission denied: {message}", error_code)
        self.operation = operation
        self.resource = resource
