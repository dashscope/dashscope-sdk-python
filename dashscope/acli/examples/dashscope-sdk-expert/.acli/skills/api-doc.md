---
name: api-doc
description: Look up SDK API parameters and usage
arguments: [api_name]
---

First use run_command to check the installed SDK version and the actual API signature:
1. `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`
2. `run_command("python -c 'import inspect, dashscope; print(inspect.signature(dashscope.{api_name}.call))'")`
3. If the signature is not clear enough, use `run_command("python -c 'import dashscope; help(dashscope.{api_name})'")` to get the full documentation.

Based on the actual inspection results, explain all parameters of {api_name}, the return value format, and the supported model list, and provide a typical usage example (Python).

Note: parameter descriptions must match the inspection results. Do not answer from memory.
