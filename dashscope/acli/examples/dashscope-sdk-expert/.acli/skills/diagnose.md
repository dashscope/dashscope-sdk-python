---
name: diagnose
description: Read a code file and diagnose SDK call errors
arguments: [file_path]
---

Diagnostic steps:

1. Use `read_file` to read the user's code at `{file_path}`. Identify the API generation from the call itself: `Generation.call` and similar module/class APIs are 1.x; `Client` / `AsyncClient` resource calls are 2.0. An installed 2.0 package does not by itself make existing 1.x code wrong.
2. Check the installed version: `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`.
3. Inspect the exact called object and method with `inspect.signature`; for 2.0, inspect the bound resource method and relevant response model. Do not assume a `.call()` method or a `.output` response envelope.
4. Classify the failure from its actual type, `code`, `status_code`, `message`, and `request_id`. For 2.0, use the `dashscope.errors` hierarchy and distinguish HTTP/API errors from `SdkError` connection, timeout, and local failures. For 1.x, preserve its response-checking and exception conventions.
5. Only if a specific behavior remains unclear, locate the package with `run_command("python -c 'import dashscope; import os; print(os.path.dirname(dashscope.__file__))'")` and read the relevant implementation.

Point out the root cause and provide complete fixed code. When repairing existing 1.x code, keep its style and contract unless the user explicitly requests migration. When writing or migrating to 2.0, use current resource methods, flattened responses, client lifecycle management, and `dashscope.errors`; do not mix 1.x and 2.0 return handling in the same example.
