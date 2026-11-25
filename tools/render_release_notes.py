# -*- coding: utf-8 -*-
"""Render the draft GitHub Release body from conventional commits.

Used by the ``release`` job in .github/workflows/release.yml: the commit
subjects between the previous tag and the released tag are grouped by
their conventional-commit type (feat/fix/refactor/docs/chore...) and
substituted into .github/release-template.md, so the draft Release ships
with a ready-made changelog instead of empty sections.

Usage (from the repo root, after ``git fetch --tags --force``):

    python tools/render_release_notes.py \
        --tag vX.Y.Z --repo owner/name --output release-body.md

Only the Highlights and Contributors sections stay manual: they are the
editorial parts a machine cannot write. Everything else is generated and
can still be edited on the draft page before publishing.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / ".github" / "release-template.md"

# Conventional-commit subject: type(scope)!: description. The optional
# ``!`` marks a breaking change; the scope is kept as a bold prefix.
_SUBJECT_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?"
    r":\s*(?P<desc>.+)$",
)

# Section order mirrors .github/release-template.md. Each entry maps a
# template placeholder to the commit types it collects; subjects that
# match no known type land in OTHER (never silently dropped).
_SECTIONS = [
    ("FEATURES", "✨ Features", ("feat",)),
    ("BUG_FIXES", "🐛 Bug Fixes", ("fix",)),
    ("REFACTORING", "♻️ Refactoring", ("refactor",)),
    ("DOCUMENTATION", "📚 Documentation", ("docs",)),
    ("CI_CHORE", "🔧 CI & Chore", ("chore", "ci", "build", "test", "style", "perf")),
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def previous_tag(tag: str) -> str:
    """Newest version-sorted tag other than ``tag`` (empty when first)."""
    tags = git("tag", "--sort=-v:refname").splitlines()
    for candidate in tags:
        if candidate and candidate != tag:
            return candidate
    return ""


def commit_messages(rev_range: str) -> list:
    """Full messages (subject + body) of non-merge commits in range."""
    out = git("log", "--no-merges", "--format=%B%x00", rev_range)
    return [msg.strip() for msg in out.split("\0") if msg.strip()]


def bullet(subject_match) -> str:
    desc = subject_match.group("desc")
    scope = subject_match.group("scope")
    return f"- **{scope}:** {desc}" if scope else f"- {desc}"


def group_commits(messages: list) -> dict:
    """Group commit messages into template sections.

    Returns {placeholder: markdown bullets or "None"} plus
    BREAKING_CHANGES. Duplicate subjects (e.g. a cherry-pick and its
    original) collapse to one bullet, keeping first-seen order.
    """
    groups = {name: [] for name, _, _ in _SECTIONS}
    groups["OTHER"] = []
    breaking = []
    seen = set()
    for message in messages:
        subject = message.splitlines()[0].strip()
        if subject in seen:
            continue
        seen.add(subject)
        match = _SUBJECT_RE.match(subject)
        if match is None:
            groups["OTHER"].append(f"- {subject}")
            continue
        kind = match.group("type").lower()
        target = "OTHER"
        for name, _, types in _SECTIONS:
            if kind in types:
                target = name
                break
        groups[target].append(bullet(match))
        if match.group("breaking") or "BREAKING CHANGE" in message:
            breaking.append(bullet(match))
    rendered = {
        name: "\n".join(bullets) if bullets else "None"
        for name, bullets in groups.items()
    }
    rendered["BREAKING_CHANGES"] = "\n".join(breaking) if breaking else "None"
    return rendered


def render_body(tag: str, changelog_url: str, groups: dict) -> str:
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    other = groups.get("OTHER", "None")
    other_section = (
        f"### 📌 Other\n\n{other}\n\n" if other != "None" else ""
    )
    return template.safe_substitute(
        TAG_NAME=tag,
        VERSION=tag.lstrip("v"),
        FULL_CHANGELOG_URL=changelog_url,
        OTHER_SECTION=other_section,
        **{name: groups[name] for name, _, _ in _SECTIONS},
        BREAKING_CHANGES=groups["BREAKING_CHANGES"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="released tag, e.g. v2.1.2")
    parser.add_argument("--repo", required=True, help="owner/name on GitHub")
    parser.add_argument("--output", required=True, help="body file to write")
    args = parser.parse_args(argv)

    prev = previous_tag(args.tag)
    if prev:
        rev_range = f"{prev}..{args.tag}"
        changelog_url = (
            f"https://github.com/{args.repo}/compare/{prev}...{args.tag}"
        )
    else:
        rev_range = args.tag
        changelog_url = f"https://github.com/{args.repo}/commits/{args.tag}"
    groups = group_commits(commit_messages(rev_range))
    body = render_body(args.tag, changelog_url, groups)
    Path(args.output).write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
