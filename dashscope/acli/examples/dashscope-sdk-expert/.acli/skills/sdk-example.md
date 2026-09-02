---
name: sdk-example
description: Generate runnable DashScope SDK code examples (Python/Java)
arguments: [api_name]
---

Before generating code, verify the user's installed SDK version and API signature:
1. `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`
2. `run_command("python -c 'import inspect, dashscope; print(inspect.signature(dashscope.{api_name}.call))'")`

Based on the actual signature, generate a directly runnable code example (Python) for the DashScope SDK's {api_name}, including imports, the call, and result handling. If both streaming and non-streaming modes are available, show both. Make sure to use the latest messages format.

Parameter names and types in the generated code must exactly match the inspection results.
