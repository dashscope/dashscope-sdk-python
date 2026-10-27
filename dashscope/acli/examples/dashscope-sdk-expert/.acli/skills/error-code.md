---
name: error-code
description: Explain a DashScope API error code and how to fix it
arguments: [code]
---

First identify whether the failing call uses the 1.x response convention or the 2.0 exception convention. Inspect the installed SDK and search for `{code}` in the public error definitions rather than guessing from the message:

1. `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`
2. `run_command("python -c 'import dashscope; import os; print(os.path.dirname(dashscope.__file__))'")`
3. Search the relevant error registry, protocol parser, or 1.x API implementation for the exact wire code. Preserve the original `code`, `message`, and `request_id` in the diagnosis.

## Python 2.0 errors

New code imports concrete exceptions from `dashscope.errors`. The package root directly exports `DashScopeError` and `ErrorCode`, not every concrete subclass.

```python
from dashscope.errors import (
    BadRequestError,
    ContentFilterError,
    DashScopeError,
    RateLimitError,
)

try:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages="Hello",
    )
except ContentFilterError as error:
    print("Content blocked", error.request_id)
except BadRequestError as error:
    print("Bad request", error.code, error.message, error.request_id)
except RateLimitError as error:
    print("Throttled", error.status_code, error.request_id)
except DashScopeError as error:
    print(type(error).__name__, error.code, error.message, error.request_id)
```

`ContentFilterError` is a `BadRequestError` subclass, so its `except` clause must come first.

Both HTTP non-2xx responses and HTTP 200 responses carrying a business error code use the same exception family. Common HTTP status classes are:

| Status | Exception |
| --- | --- |
| 400 | `BadRequestError` |
| 401 | `AuthenticationError` |
| 403 | `PermissionDeniedError` |
| 404 | `NotFoundError` |
| 413 | `RequestTooLargeError` |
| 429 | `RateLimitError` |
| 500 | `InternalServerError` |
| 503 | `ServiceUnavailableError` |
| 504 | `GatewayTimeoutError` |

Wire codes refine the meaning: for example, `Arrearage` maps to `PermissionDeniedError`, not 429, and throttling codes map to `RateLimitError`. A gateway timeout is `GatewayTimeoutError`; do not label every unknown 5xx as `InternalServerError` without checking the actual status and code. `ResponseError` concerns response parsing and is not synonymous with every HTTP 500.

Exceptions expose `message`, `code`, `status_code`, `request_id`, `body`, and `headers` as applicable. Connection failures, timeouts, WebSocket failures, and SDK-local failures belong to the `SdkError` branch; they have no HTTP response, have `status_code is None`, and may also provide `solution` and `allow_retry`. Do not discard these fields by flattening every failure into a status-only message.

With `raise_on_error=True` (the default), catch `DashScopeError` or a more specific subclass. If the user explicitly selects return-value mode, check `is_error`:

```python
lenient = client.with_options(raise_on_error=False)
response = lenient.chat.completions.create(model="qwen-plus", messages="Hello")
if response.is_error:
    print(response.code, response.message, response.request_id)
```

`raise_on_error=False` does not suppress connection failures or timeouts because no response object exists. In a stream, a business error may be represented by an `is_error=True` item before iteration ends. Automatic transport retries do not make authentication, permission, quota, or parameter errors safe to retry blindly.

## Python 1.x and compatible-mode errors

For an existing 1.x call such as `Generation.call`, preserve its contract: API failures commonly return a response whose `status_code`, `code`, `message`, and `request_id` must be checked, while local validation errors may still be raised. Do not convert this into a 2.0 `except DashScopeError` example unless the user is migrating.

OpenAI-compatible code using the official `openai` package keeps that library's own exception hierarchy; it is not the `dashscope.errors` hierarchy. Java errors likewise follow the Java SDK's independent API and version line.

Based on the verified implementation, explain what `{code}` means, common triggers, whether retry is safe, and concrete fix steps with an example in the user's existing SDK style.
