---
name: api-doc
description: Look up SDK API parameters and usage
arguments: [api_name]
---

First identify whether `{api_name}` refers to a 1.x module/class API or a 2.0 client resource, then inspect the installed SDK:

1. `run_command("python -c 'import dashscope; print(dashscope.__version__)'")`.
2. Resolve the exact public object and method. Use `inspect.signature` on that bound method; use `help()` on the resource or response model when parameter or return-field documentation is insufficient. Do not mechanically append `.call` to `{api_name}`.
3. For a 2.0 resource, a local `Client(api_key="sk-test")` is enough for inspection and does not make a network request. Close it afterwards. Read the resource implementation or Pydantic response model only when inspection leaves a specific question unresolved.
4. Consult the matching domain skill for the current 2.0.3 mapping and the 1.x compatibility notes.

Explain the parameters, return type and fields, streaming/task lifecycle where applicable, and supported models from the verified implementation. Include a typical Python example and distinguish these contracts explicitly:

- 2.0: `Client` / `AsyncClient` resource methods, flattened typed responses, `dashscope.errors` exceptions, and `jobs.create/fetch/wait` for supported long-running tasks.
- 1.x: the existing `.call()` signature and its actual response or task fields; do not rewrite them as 2.0 fields.

Parameter descriptions and model support must match the inspection results. If the installed version differs from the knowledge base, state the difference instead of guessing.
