---
description: "About the DashScope Python SDK: project links, Apache-2.0 license, compatibility and versioning policy."
---
# About

## The project

The DashScope Python SDK (`dashscope` on PyPI) is the official Python client
for [Alibaba Cloud Model Studio (Bailian)](https://www.alibabacloud.com/help/en/model-studio/),
covering text generation, multimodal understanding, embeddings, reranking,
image/video generation, speech synthesis & recognition, fine-tuning,
agentic RL, managed agents and applications.

- Source code: [github.com/dashscope/dashscope-sdk-python](https://github.com/dashscope/dashscope-sdk-python)
- Package: [pypi.org/project/dashscope](https://pypi.org/project/dashscope/)
- Service documentation: [Model Studio help center](https://www.alibabacloud.com/help/en/model-studio/)

## License

The SDK is licensed under the
[Apache License 2.0](https://github.com/dashscope/dashscope-sdk-python/blob/main/LICENSE).
Third-party notices (e.g. the bundled certifi CA bundle) are listed in
[NOTICE](https://github.com/dashscope/dashscope-sdk-python/blob/main/NOTICE).

## Compatibility and versioning

- The 2.x line is the actively developed API: resource-style `Client` /
  `AsyncClient`, typed responses and exceptions.
- The complete 1.x public surface remains importable from its historical
  paths and is implemented by frozen facades under `dashscope.legacy_1x`;
  it receives no new features.
- AgentStudio and Agentic RL are application sub-SDKs with their own
  client and typed returns (AgentStudio also has dedicated resources and
  pagination); their errors build on the unified `dashscope.errors`
  hierarchy: AgentStudio raises the main SDK exceptions directly (legacy
  names like `AgentStudioError` are aliases), while Agentic RL keeps
  dedicated `AgenticRLError` subclasses within that same hierarchy.
- Python 3.9+ is required; the optional interactive assistant
  (`dashscope[expert]`) requires Python 3.10+.
- Current SDK version: **{{ sdk_version }}** — see
  [Release Notes](release-notes.md).

## Getting help

- Report bugs and request features in
  [GitHub Issues](https://github.com/dashscope/dashscope-sdk-python/issues).
- Browse the [API Reference](reference/client.md) or the
  [Quickstart](quickstart.md).
- Failed calls can explain themselves: enable the devkit diagnosis to get
  structured fix guidance on `DashScopeError.diagnosis`
  (see [Devkit](reference/devkit.md)).