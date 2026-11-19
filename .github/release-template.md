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

-

### ✨ Features

-

### ♻️ Refactoring

-

### 🐛 Bug Fixes

-

### 📚 Documentation

-

### 🔧 CI & Chore

-

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

-

### 📝 Full Changelog

${FULL_CHANGELOG_URL}