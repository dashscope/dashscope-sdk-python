## ${TAG_NAME} Release Notes

<!--
Maintainer guide (please read before publishing; this HTML comment is
invisible in the rendered Release page):

1. This file is the skeleton for the draft Release body. It is rendered
   by the `release` job in .github/workflows/release.yml using Python's
   string.Template.safe_substitute.
2. Placeholders that get substituted automatically at render time:
     TAG_NAME              current tag, e.g. vX.Y.Z
     VERSION               version without the leading `v`, e.g. X.Y.Z
     FULL_CHANGELOG_URL    compare URL against the previous tag
   Note: these three names must appear as `$` followed by braces in the
   template body to be substituted; this comment intentionally spells
   them without that syntax so it does not substitute itself.
3. HTML comments under each section are classification hints. Delete or
   keep them before publishing (kept comments stay invisible to readers
   and save effort on the next release).
4. If a section has no content for this release, delete it entirely or
   write "None".
5. Classification follows conventional-commit PR title prefixes:
     feat(scope):         -> ✨ Features
     fix(scope):          -> 🐛 Bug Fixes
     refactor(scope):     -> ♻️ Refactoring
     docs(scope):         -> 📚 Documentation
     chore/ci/build/test: -> 🔧 CI & Chore
6. Highlights is the "elevator pitch" — pick 1-3 changes users should
   care about most (new capabilities, breaking changes, or fixes with
   wide impact).
7. For pre-release versions (tag contains a/b/rc/post), tick
   "Set as a pre-release" on the draft page so this release does not
   occupy the /releases/latest endpoint.
-->

### 🚀 Highlights

- Qwen Live realtime audio sessions built on a shared realtime transport.
- New `dashscope.devkit`: offline self-diagnosis for failed calls plus static/record/replay HTTP mocking.
- Automated release pipeline — pushing a `v*` tag now builds, publishes to PyPI, and drafts the GitHub Release.

### ✨ Features

- audio: add Qwen Live sessions on the shared realtime transport (#231).
- audio: accept audio sources for voice enrollment.
- agentstudio: session mount fields for Bailian memory and OSS.
- devkit: add offline self-diagnosis for failed calls.
- devkit: add static/record/replay dev mock at the HTTP transport.
- transport: introduce `APIResponse` and a `raw` parameter for raw responses.
- transport: support per-request options across all resources.
- pagination: introduce a generic pagination base class and refactor resource list interfaces.
- parse: add fault-tolerant response model parsing and strict validation config.
- core: add client request ID and response validation exception.
- v2: extend typed parsing, per-request options, and pagination fixes to all resources.
- Enhance API response handling and resource methods.

### ♻️ Refactoring

- cli: migrate the agent runtime to `dashscope-agenticx` (#230).
- server: extract client disconnect response logic.

### 🐛 Bug Fixes

- tts: prevent `KeyError` for non-timestamp messages in speech parts.
- audio: preserve graceful Live and Realtime shutdown.
- reinforcement: preserve FaaS business errors and request context.
- reinforcement: address agentic RL review findings.
- reinforcement: allow local wheel paths in pip requirement tokens.
- core: unify API key resolution and improve error handling.
- v2: support more image models and input forms, including wan2.7 image responses.
- logging: correlate SDK 2.0 response errors with request IDs.

### 📚 Documentation

- None

### 🔧 CI & Chore

- release: add tag-triggered PyPI publishing workflow (#234).
- version: bump to 2.1.0.
- ci: pin Python 3.9 / unit-test runners to ubuntu-24.04.
- ci: run legacy unit tests on pushes and pull requests again.
- test: stabilize RL CLI help checks, align job list output, and move layer decoupling checks to the current suite.

### ⚠️ Breaking Changes

None

---

### 📦 Installation

```bash
pip install --upgrade dashscope==${VERSION}
```

### 👥 Contributors

Thanks to all the contributors who made this release possible:

<!-- List @handles here. On the draft page, the "Generate release notes"
     button in the top-right can auto-populate New Contributors. -->

- @zhansheng.lzs
- @luk384090-cloud
- @songguocola
- @foleydang
- @lzsweb

### 📝 Full Changelog

${FULL_CHANGELOG_URL}