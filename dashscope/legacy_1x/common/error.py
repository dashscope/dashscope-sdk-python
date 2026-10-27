# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The 1.x exception family (pending deprecation).

Every class here belongs to the 1.x API surface and is *going to be*
deprecated along with it: instantiation emits a
:exc:`PendingDeprecationWarning` (ignored by Python's default filters,
so current 1.x users see no noise), and static checkers flag the names
via ``__deprecated__``. New code should use the 2.0 family in
:mod:`dashscope.errors` (``DashScopeError`` and its subclasses). The
1.x classes stay fully supported and importable so existing ``except``
clauses keep working.

``DashScopeException`` carries no marker at all: it is re-exported from
the neutral layer (:mod:`dashscope._shared._errors`) that both stacks
inherit from.
"""

from typing_extensions import deprecated

# The base class lives in the neutral layer so the AgentStudio sub-SDK
# (owned by the 2.0 package) and the 1.x classes below inherit from one
# and the same object.
from dashscope._shared._errors import DashScopeException


def _legacy_error(name):
    """Pending-deprecation marker for one 1.x error class.

    Warns on instantiation (and subclasses), never on import, so the
    SDK's own 1.x modules can keep importing these names silently.
    """
    return deprecated(
        f"dashscope.common.error.{name} is part of the 1.x API, which "
        f"is scheduled for deprecation and will be removed in a future "
        f"release; use the dashscope.errors (2.0) exception family "
        f"instead.",
        category=PendingDeprecationWarning,
    )


@_legacy_error("AuthenticationError")
class AuthenticationError(DashScopeException):
    pass


@_legacy_error("InvalidParameter")
class InvalidParameter(DashScopeException):
    pass


@_legacy_error("InvalidTask")
class InvalidTask(DashScopeException):
    pass


@_legacy_error("UnsupportedModel")
class UnsupportedModel(DashScopeException):
    pass


@_legacy_error("UnsupportedTask")
class UnsupportedTask(DashScopeException):
    pass


@_legacy_error("ModelRequired")
class ModelRequired(DashScopeException):
    pass


@_legacy_error("InvalidModel")
class InvalidModel(DashScopeException):
    pass


@_legacy_error("InvalidInput")
class InvalidInput(DashScopeException):
    pass


@_legacy_error("InvalidFileFormat")
class InvalidFileFormat(DashScopeException):
    pass


@_legacy_error("UnsupportedApiProtocol")
class UnsupportedApiProtocol(DashScopeException):
    pass


@_legacy_error("NotImplemented")
class NotImplemented(DashScopeException):  # pylint: disable=redefined-builtin
    pass


@_legacy_error("MultiInputsWithBinaryNotSupported")
class MultiInputsWithBinaryNotSupported(DashScopeException):
    pass


@_legacy_error("UnexpectedMessageReceived")
class UnexpectedMessageReceived(DashScopeException):
    pass


@_legacy_error("UnsupportedData")
class UnsupportedData(DashScopeException):
    pass


@_legacy_error("AssistantError")
class AssistantError(DashScopeException):
    def __init__(self, **kwargs):
        self.message = None
        self.code = None
        self.request_id = None
        if "message" in kwargs:
            import json

            msg = json.loads(kwargs["message"])
            if "request_id" in msg:
                self.request_id = msg["request_id"]
            if "code" in msg:
                self.code = msg["code"]
            if "message" in msg:
                self.message = msg["message"]

    def __str__(self):
        msg = (
            f"Request failed, request_id: {self.request_id}, "
            f"code: {self.code}, message: {self.message}"
        )
        return msg


# for server send generation or inference error.
@_legacy_error("RequestFailure")
class RequestFailure(DashScopeException):
    def __init__(
        self,
        request_id=None,
        message=None,
        name=None,
        http_code=None,
    ):
        self.request_id = request_id
        self.message = message
        self.name = name
        self.http_code = http_code

    def __str__(self):
        # pylint: disable=line-too-long
        msg = (
            f"Request failed, request_id: {self.request_id}, "
            f"http_code: {self.http_code} error_name: {self.name}, "
            f"error_message: {self.message}"
        )
        return msg


@_legacy_error("UnknownMessageReceived")
class UnknownMessageReceived(DashScopeException):
    pass


@_legacy_error("InputDataRequired")
class InputDataRequired(DashScopeException):
    pass


@_legacy_error("InputRequired")
class InputRequired(DashScopeException):
    pass


@_legacy_error("UnsupportedDataType")
class UnsupportedDataType(DashScopeException):
    pass


@_legacy_error("ServiceUnavailableError")
class ServiceUnavailableError(DashScopeException):
    pass


@_legacy_error("UnsupportedHTTPMethod")
class UnsupportedHTTPMethod(DashScopeException):
    pass


@_legacy_error("AsyncTaskCreateFailed")
class AsyncTaskCreateFailed(DashScopeException):
    pass


@_legacy_error("UploadFileException")
class UploadFileException(DashScopeException):
    pass


@_legacy_error("TimeoutException")
class TimeoutException(DashScopeException):
    pass
