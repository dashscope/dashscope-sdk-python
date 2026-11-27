---
description: "API reference for the DashScopeError exception hierarchy and error codes."
---
# Errors

All SDK exceptions inherit from `DashScopeError` and carry `message`, `code`,
`status_code`, `request_id`, `body` and `headers`. Catch the base class to
handle any SDK or API failure, or catch specific subclasses for finer control.

## Base class & error codes

::: dashscope.errors.DashScopeError

::: dashscope.errors.ErrorCode

::: dashscope.errors.SdkError

## Transport & validation errors

::: dashscope.errors.APIConnectionError

::: dashscope.errors.APITimeoutError

::: dashscope.errors.APIResponseValidationError

## HTTP status errors

::: dashscope.errors.APIStatusError

::: dashscope.errors.BadRequestError

::: dashscope.errors.AuthenticationError

::: dashscope.errors.PermissionDeniedError

::: dashscope.errors.NotFoundError

::: dashscope.errors.RequestTooLargeError

::: dashscope.errors.RateLimitError

::: dashscope.errors.InternalServerError

::: dashscope.errors.ServiceUnavailableError

::: dashscope.errors.GatewayTimeoutError

::: dashscope.errors.ContentFilterError

## WebSocket errors

::: dashscope.errors.WsEventError

::: dashscope.errors.WsStateError

::: dashscope.errors.WsQueueCapacityExceededError

::: dashscope.errors.WsBinaryOwnerConflictError
