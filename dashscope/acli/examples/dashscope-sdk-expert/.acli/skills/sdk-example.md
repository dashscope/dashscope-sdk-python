---
name: sdk-example
description: Generate runnable DashScope SDK code examples (Python/Java)
arguments: [api_name]
---

Before generating code, identify the API generation and verify the user's installed SDK:

1. Check the version: `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`.
2. Treat resource calls such as `client.chat.completions.create` as 2.0 and module/class calls such as `dashscope.Generation.call` as 1.x. Do not infer a migration request merely because 2.0 is installed.
3. Resolve the actual object named by `{api_name}`, then inspect the real method with `inspect.signature` and, when needed, `help()`. For a 2.0 resource, instantiate `dashscope.Client(api_key="sk-test")`, inspect the bound resource method, and close the client; do not assume every API has a `.call()` method.
4. Cross-check the matching domain skill and cookbook example. Parameter names and types must match the installed implementation, not memory.

Generate a directly runnable Python example for `{api_name}` with imports, invocation, response handling, and error handling.

- For new 2.0 code, use `Client` or `AsyncClient`, close it directly or with a context manager, and read the flattened typed response rather than `.output`. Show streaming and non-streaming forms only when that resource supports both; chat streams expose incremental `delta` content.
- For a fix to existing 1.x code, preserve its `.call()` style and response contract unless the user explicitly asks to migrate. Keep response `status_code` / `code` / `message` checks where they belong.
- Mark capabilities that exist only in 2.0. Do not invent compatibility aliases, resources, parameters, or return fields.

Keep Java examples on the Java SDK's own version line; do not substitute Python 2.0 signatures for Java APIs.
