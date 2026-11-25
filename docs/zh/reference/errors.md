---
description: "DashScopeError 异常层级与错误码 API 参考。"
---
# 错误（Errors）

SDK 的所有异常都继承自 `DashScopeError`，携带 `message`、`code`、
`status_code`、`request_id`、`body` 与 `headers`。捕获基类即可处理任何
SDK 或 API 失败，也可捕获具体子类做更精细的控制。

## 基类与错误码

::: dashscope.errors.DashScopeError

::: dashscope.errors.ErrorCode

::: dashscope.errors.SdkError

## 传输与校验错误

::: dashscope.errors.APIConnectionError

::: dashscope.errors.APITimeoutError

::: dashscope.errors.APIResponseValidationError

## HTTP 状态错误

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

## WebSocket 错误

::: dashscope.errors.WsEventError

::: dashscope.errors.WsStateError

::: dashscope.errors.WsQueueCapacityExceededError

::: dashscope.errors.WsBinaryOwnerConflictError
