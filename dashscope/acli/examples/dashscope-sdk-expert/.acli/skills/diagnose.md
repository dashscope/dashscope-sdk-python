---
name: diagnose
description: Read a code file and diagnose SDK call errors
arguments: [file_path]
---

Diagnostic steps:
1. Use `read_file` to read the user's code at {file_path}.
2. Check the installed SDK version: `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`
3. If the code calls a specific API (e.g. `dashscope.Generation.call`), inspect its actual signature: `run_command("python -c 'import inspect, dashscope; print(inspect.signature(dashscope.Generation.call))'")`
4. If you need to understand SDK internals, locate the source with `run_command("python -c 'import dashscope; import os; print(os.path.dirname(dashscope.__file__))'")`, then use `read_file` to read the relevant implementation.

Based on the actual findings above, point out the problems in the user's code and provide the complete fixed code.
