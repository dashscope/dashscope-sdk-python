# -*- coding: utf-8 -*-
"""
Custom exception hierarchy for the AgenticRL system
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from dashscope.common.error_registry import PublicError


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


def _build_safe_fallback(error_code: int, message: str) -> Exception:
    """Build a safe public exception when no public_error mapping exists.

    Logs the internal error_code for backend debugging and returns a
    generic InternalServerError so that integer error codes are never
    exposed to SDK callers.
    """
    from dashscope.finetune.reinforcement import logger
    from dashscope.common.error_registry import INTERNAL_ERROR
    from dashscope.common.error import DashScopeException

    logger.error(
        "AgenticRL internal error (no public_error mapping): code=%s, "
        "message=%s",
        error_code,
        message,
        exc_info=True,
    )

    formatted_msg = INTERNAL_ERROR.format_msg()
    safe_exc = DashScopeException(formatted_msg)
    safe_exc.status_code = INTERNAL_ERROR.status_code
    safe_exc.error_code = INTERNAL_ERROR.error_code
    safe_exc.request_id = None  # type: ignore[attr-defined]
    return safe_exc


# Module-level mapping from PublicError.error_code (string class name) to
# SDK exception class. Built once at first use; ImportError is logged as a
# warning so that refactoring in error.py is never silently swallowed.
_ERROR_CODE_TO_CLASS: Dict[str, type] = {}
_ERROR_CODE_TO_CLASS_LOADED = False


def _load_error_code_mapping() -> None:
    """Lazily load the error-code-to-class mapping with ImportError guard."""
    global _ERROR_CODE_TO_CLASS, _ERROR_CODE_TO_CLASS_LOADED
    if _ERROR_CODE_TO_CLASS_LOADED:
        return

    from dashscope.common.error import DashScopeException

    try:
        from dashscope.common.error import (
            AuthenticationError,
            InvalidParameter,
            ServiceUnavailableError,
            TimeoutException,
        )

        _ERROR_CODE_TO_CLASS.update(
            {
                "AuthenticationError": AuthenticationError,
                "BadRequestError": InvalidParameter,
                "PermissionDeniedError": DashScopeException,
                "NotFoundError": DashScopeException,
                "RequestTooLargeError": DashScopeException,
                "RateLimitError": DashScopeException,
                "InternalServerError": DashScopeException,
                "ServiceUnavailableError": ServiceUnavailableError,
                "GatewayTimeoutError": TimeoutException,
            },
        )
    except ImportError:
        from dashscope.finetune.reinforcement import logger

        logger.warning(
            "Failed to import SDK exception classes from "
            "dashscope.common.error; all AgenticRL errors will fall back "
            "to generic DashScopeException.",
            exc_info=True,
        )

    _ERROR_CODE_TO_CLASS_LOADED = True


def _convert_with_public_error(
    public_error: "PublicError",
    internal_error_code: int,
    internal_message: str,
    timestamp: str = "",
) -> Exception:
    """Convert an internal error to a public SDK exception via public_error.

    Handles two failure modes safely:
    1. ImportError of SDK exception classes → logged as warning, falls back
       to generic DashScopeException.
    2. public_error.format_msg() raises → caught and replaced with a
       hard-coded safe message to prevent leaking internal details.
    """
    from dashscope.finetune.reinforcement import logger
    from dashscope.common.error import DashScopeException

    # Log internal error details for backend debugging
    logger.error(
        "AgenticRL internal error: code=%s, message=%s%s",
        internal_error_code,
        internal_message,
        f", timestamp={timestamp}" if timestamp else "",
        exc_info=True,
    )

    _load_error_code_mapping()

    exc_class = _ERROR_CODE_TO_CLASS.get(
        public_error.error_code,
        DashScopeException,
    )

    # Guard format_msg() to prevent leaking internal details on failure
    try:
        formatted_msg = public_error.format_msg()
    except Exception:
        logger.warning(
            "public_error.format_msg() failed for error_code=%s; "
            "using safe fallback message.",
            public_error.error_code,
            exc_info=True,
        )
        formatted_msg = "An internal error occurred. Please try again later."

    exc = exc_class(formatted_msg)
    exc.status_code = public_error.status_code  # type: ignore[attr-defined]
    exc.error_code = public_error.error_code  # type: ignore[attr-defined]
    exc.request_id = None  # type: ignore[attr-defined]
    return exc


class AgenticRLError(_RootCauseMixin, Exception):
    """Base class for all Agentic RL exceptions."""

    def __init__(
        self,
        message: str,
        error_code: int = 1000,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.public_error = public_error
        self.timestamp = datetime.now().isoformat()
        self.message = message

    def __str__(self):
        base = f"[{self.error_code}] {self.message} (at {self.timestamp})"
        return f"{base}{self._format_cause()}"

    def to_public_exception(self) -> Exception:
        """Convert to a standard SDK exception aligned with error_registry."""
        if self.public_error is None:
            return _build_safe_fallback(self.error_code, self.message)
        return _convert_with_public_error(
            self.public_error,
            self.error_code,
            self.message,
            self.timestamp,
        )


class IOErrorWithCode(AgenticRLError):
    """Raised for general I/O operation failures"""

    def __init__(
        self,
        message: str,
        error_code: int = 1800,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(f"I/O error: {message}", error_code, public_error)
        self.path = path
        self.operation = operation


class RuntimeErrorWithCode(_RootCauseMixin, RuntimeError):
    """Enhanced RuntimeError that supports error codes for better error
    categorization."""

    def __init__(
        self,
        message: str,
        error_code: int = 0,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.public_error = public_error
        self.message = message

    def __str__(self):
        return f"[{self.error_code}] {self.message}{self._format_cause()}"

    def to_public_exception(self) -> Exception:
        """Convert to a standard SDK exception aligned with error_registry."""
        if self.public_error is None:
            return _build_safe_fallback(self.error_code, self.message)
        return _convert_with_public_error(
            self.public_error,
            self.error_code,
            self.message,
        )


class ValueErrorWithCode(_RootCauseMixin, ValueError):
    """Enhanced ValueError that supports error codes for better error
    categorization."""

    def __init__(
        self,
        message: str,
        error_code: int = 0,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.public_error = public_error
        self.message = message

    def __str__(self):
        return f"[{self.error_code}] {self.message}{self._format_cause()}"

    def to_public_exception(self) -> Exception:
        """Convert to a standard SDK exception aligned with error_registry."""
        if self.public_error is None:
            return _build_safe_fallback(self.error_code, self.message)
        return _convert_with_public_error(
            self.public_error,
            self.error_code,
            self.message,
        )


class InputError(AgenticRLError):
    """Raised when invalid input data is detected during validation"""

    def __init__(
        self,
        message: str,
        error_code: int = 1100,
        field: Optional[str] = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message, error_code, public_error)
        self.field = field


class OutputError(AgenticRLError):
    """Raised when service response fails output validation checks"""

    def __init__(
        self,
        message: str,
        error_code: int = 1200,
        response: Optional[Dict] = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message, error_code, public_error)
        self.response = response


class BaseConnectionError(AgenticRLError):
    """Base class for connection-related errors"""

    def __init__(
        self,
        message: str,
        error_code: int = 1300,
        endpoint: Optional[str] = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message, error_code, public_error)
        self.endpoint = endpoint


class OSSConnectionError(BaseConnectionError):
    """Raised when connecting to OSS storage service fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1310,
        endpoint: str = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"OSS connection failed: {message}",
            error_code,
            endpoint,
            public_error,
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
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"OSS upload failed: {message}",
            error_code,
            endpoint,
            public_error,
        )
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
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(message, error_code, public_error)
        self.resource_id = resource_id


class RegistrationError(DeploymentError):
    """Raised when function registration fails in the deployment system"""

    def __init__(
        self,
        message: str,
        error_code: int = 1410,
        resource_id: Optional[str] = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Registration failed: {message}",
            error_code,
            resource_id,
            public_error,
        )


class DatasetsError(DeploymentError):
    """Raised when update datasets fails in the deployment system"""

    def __init__(
        self,
        message: str,
        error_code: int = 1460,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Datasets failed: {message}",
            error_code,
            public_error=public_error,
        )


class FunctionLoadError(DeploymentError):
    """Raised when loading a registered function into runtime fails"""

    def __init__(
        self,
        message: str,
        error_code: int = 1420,
        entity_id: str = None,
        error_log: Optional[str] = None,
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Function load failed: {message}",
            error_code,
            entity_id,
            public_error,
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
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Function layer create failed: {message}",
            error_code,
            layer_name,
            public_error,
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
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Instance warmup failed: {message}",
            error_code,
            public_error=public_error,
        )
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
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Instance query failed: {message}",
            error_code,
            public_error=public_error,
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
        public_error: Optional["PublicError"] = None,
    ):
        super().__init__(
            f"Validation failed: {message}",
            error_code,
            public_error,
        )
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
