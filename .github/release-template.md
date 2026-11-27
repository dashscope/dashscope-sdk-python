## ${TAG_NAME} Release Notes

<!--
Maintainer guide (please read before publishing; this HTML comment is
invisible in the rendered Release page):

1. This file is the skeleton for the draft Release body. It is rendered
   by tools/render_release_notes.py (invoked by the `release` job in
   .github/workflows/release.yml) using Python's
   string.Template.safe_substitute.
2. Placeholders that get substituted automatically at render time:
     TAG_NAME              current tag, e.g. vX.Y.Z
     VERSION               version without the leading `v`, e.g. X.Y.Z
     FULL_CHANGELOG_URL    compare URL against the previous tag
     FEATURES / BUG_FIXES / REFACTORING / DOCUMENTATION / CI_CHORE
                           bullet lists grouped from the conventional
                           commit subjects between the previous tag and
                           this tag (empty sections render as "None")
     BREAKING_CHANGES      commits marked `type!:` or carrying a
                           BREAKING CHANGE footer (else "None")
     OTHER_SECTION         a full "Other" section for subjects with no
                           recognized type; empty string when none
   Note: these names must appear as `$` followed by braces in the
   template body to be substituted; this comment intentionally spells
   them without that syntax so it does not substitute itself.
3. The generated lists are a starting point: edit, reorder or trim them
   on the draft page before publishing. Classification follows
   conventional-commit prefixes:
     feat(scope):         -> ✨ Features
     fix(scope):          -> 🐛 Bug Fixes
     refactor(scope):     -> ♻️ Refactoring
     docs(scope):         -> 📚 Documentation
     chore/ci/build/test/style/perf: -> 🔧 CI & Chore
4. Highlights is the "elevator pitch" — pick 1-3 changes users should
   care about most (new capabilities, breaking changes, or fixes with
   wide impact). It stays manual by design.
5. For pre-release versions (tag contains a/b/rc/post), tick
   "Set as a pre-release" on the draft page so this release does not
   occupy the /releases/latest endpoint.
-->

### 🚀 Highlights

-

### ✨ Features

${FEATURES}

### ♻️ Refactoring

${REFACTORING}

### 🐛 Bug Fixes

${BUG_FIXES}

### 📚 Documentation

${DOCUMENTATION}

### 🔧 CI & Chore

${CI_CHORE}

${OTHER_SECTION}### ⚠️ Breaking Changes

${BREAKING_CHANGES}

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