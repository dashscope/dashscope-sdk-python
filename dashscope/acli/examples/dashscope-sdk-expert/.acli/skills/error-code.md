---
name: error-code
description: Explain a DashScope API error code and how to fix it
arguments: [code]
---

First check how this error code is actually handled in the SDK:
1. `run_command("python -c 'import dashscope; import os; print(os.path.dirname(dashscope.__file__))'")`
2. Use `run_command("grep -rn '{code}' <sdk_path>")` to search the SDK source for where this error code is defined and handled.

Based on the actual code, explain what DashScope API error code {code} means, list common triggers, and give concrete fix steps with code examples.
