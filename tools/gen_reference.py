# -*- coding: utf-8 -*-
"""Keep the MkDocs API reference in sync with the package's public API.

The reference pages (docs/{en,zh}/reference/*.md) are curated prose plus
``::: dashscope...`` mkdocstrings directives. This script preserves all
prose and the existing entry order, and only:

- appends newly discovered public API symbols of an already-referenced
  module (after that module's last entry),
- adds a new section for public modules that no page covers yet
  (ownership is decided by the longest matching package prefix below),
- drops entries whose target no longer exists (renamed/removed API).

Publication contract (what counts as "public API" for discovery):

- a module defining ``__all__`` auto-grows with the classes/functions
  that are both *defined* in it and listed in ``__all__``; re-exports
  (e.g. the agentstudio root or the RL compatibility surface) stay
  curated by hand;
- a module without ``__all__`` only auto-grows when some page already
  references it, and then only with classes defined in it (public
  helper functions stay curated);
- brand-new modules need ``__all__`` to be picked up — one line that
  doubles as the module's export declaration.

Blind-spot guard: a public module that is neither referenced by any page
nor classified as internal (INTERNAL_MODULES / INTERNAL_PREFIXES) and
lacks ``__all__`` makes --check fail. Every new module must therefore be
explicitly published (``__all__``) or declared internal — it cannot slip
through silently.

The frozen ``legacy_1x`` tree gets resolution checks only, no discovery.
The ``dashscope`` root package is excluded from discovery: its exports
are documented across the individual pages, not on client.md.

Usage:
    python tools/gen_reference.py          # rewrite pages in place
    python tools/gen_reference.py --check  # CI guard: fail on drift

The docs workflow runs this before ``mkdocs build`` so the deployed
site always covers the current public API; the pytest guard
(tests/test_docs_reference.py) fails PRs whose pages are out of sync.
"""
import argparse
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Running as `python tools/gen_reference.py` puts tools/ on sys.path,
# not the repo root; make the checkout importable regardless of cwd.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REFERENCE_DIRS = {
    "en": REPO_ROOT / "docs" / "en" / "reference",
    "zh": REPO_ROOT / "docs" / "zh" / "reference",
}

# Page -> package subtrees it owns. A newly discovered module belongs to
# the page with the longest matching prefix (chat/audio win over the
# broader resources page). legacy.md is frozen: empty tuple = no walk.
PAGE_SUBTREES = {
    "client.md": ("dashscope.client", "dashscope._config"),
    "chat.md": ("dashscope.resources.chat",),
    "audio.md": ("dashscope.resources.audio",),
    "resources.md": ("dashscope.resources",),
    "protocols.md": ("dashscope.protocols",),
    "agentstudio.md": ("dashscope.applications.agentstudio",),
    "agentic-rl.md": ("dashscope.applications.finetune",),
    "types.md": ("dashscope.types",),
    "errors.md": ("dashscope.errors",),
    "devkit.md": ("dashscope.devkit",),
    "legacy.md": (),
}

# Root exports are spread across the reference pages by hand; discovering
# them here would dump the whole SDK surface onto client.md.
EXCLUDE_MODULES = {"dashscope"}

# Internal implementation subtrees that happen to export public names
# (RL tracing decorators); they are not part of the documented surface.
EXCLUDE_PREFIXES = (
    "dashscope.applications.finetune.reinforcement.component.observability",
)

# Frozen API surface: entries are resolution-checked but never grown.
FROZEN_PAGES = {"legacy.md"}

# Internal implementation modules whose public face is curated elsewhere
# (re-export facades, plumbing). A new public module must either declare
# __all__ or be listed here; --check fails on unclassified ones.
INTERNAL_MODULES = {
    "dashscope.applications.finetune",
    "dashscope.applications.finetune.reinforcement.component.func_decorator",
    "dashscope.applications.finetune.reinforcement.component.func_manager",
    "dashscope.resources",
}
INTERNAL_PREFIXES = (
    "dashscope.applications.finetune.reinforcement.common",
    "dashscope.applications.finetune.reinforcement.component.data",
    "dashscope.applications.finetune.reinforcement.component.demo",
    "dashscope.applications.finetune.reinforcement.component.parser",
    "dashscope.applications.finetune.reinforcement.component.processor",
    "dashscope.applications.finetune.reinforcement.component.server",
    "dashscope.applications.agentstudio.client",
    "dashscope.applications.agentstudio.constants",
    "dashscope.applications.agentstudio.exceptions",
    "dashscope.applications.agentstudio.pagination",
    "dashscope.applications.agentstudio.streaming",
    "dashscope.applications.agentstudio.transport",
    "dashscope.applications.agentstudio.types",
)

# Entries/modules that failed to import during the last run: kept (an
# optional dependency may be missing) but reported for manual review.
IMPORT_WARNINGS = []


def _is_internal(name: str) -> bool:
    return name in INTERNAL_MODULES or any(
        name == p or name.startswith(p + ".") for p in INTERNAL_PREFIXES
    )


def owner_page(module_name: str):
    """Page owning ``module_name`` via longest prefix match (or None)."""
    best, best_len = None, -1
    for page, subtrees in PAGE_SUBTREES.items():
        for prefix in subtrees:
            if module_name == prefix or module_name.startswith(prefix + "."):
                if len(prefix) > best_len:
                    best, best_len = page, len(prefix)
    return best


def resolve(target: str):
    """Resolve a ``:::`` target. Returns (obj, status).

    status: "ok", "missing_attr" (parent imports, attribute gone) or
    "import_error" (module itself fails to import — kept, only warned,
    so an optional dependency cannot silently delete curated entries).
    """
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        name = ".".join(parts[:i])
        try:
            obj = importlib.import_module(name)
        except ImportError:
            continue
        except Exception:
            return None, "import_error"
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None, "missing_attr"
        return obj, "ok"
    return None, "import_error"


def entry_module(target: str, obj) -> str:
    """Module an entry belongs to (whole-module entries name themselves)."""
    if inspect.ismodule(obj):
        return target
    return target.rsplit(".", 1)[0]


def public_defined(module, include_functions=True):
    """Public classes/functions defined in ``module``, in source order."""
    found = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (isinstance(obj, type)
                or (include_functions and inspect.isfunction(obj))):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if "<locals>" in getattr(obj, "__qualname__", ""):
            continue
        found.append(name)
    return found


def walk_subtree(prefix: str):
    """Public modules under a package prefix (private leaves skipped)."""
    try:
        pkg = importlib.import_module(prefix)
    except Exception as exc:  # optional dependency missing
        print(f"warning: cannot import {prefix}: {exc}", file=sys.stderr)
        return []
    if not hasattr(pkg, "__path__"):
        return [prefix]
    names = [prefix]
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        names.append(info.name)
    return names


def parse_entries(lines):
    """[(line_index, target)] for every ``:::`` directive line."""
    entries = []
    for i, line in enumerate(lines):
        if line.startswith("::: "):
            entries.append((i, line[4:].strip().split()[0]))
    return entries


def sync_page(text: str, page: str, extra_covered=frozenset()):
    """Return (new_text, additions, removals) for one reference page.

    ``extra_covered`` carries canonical identities already documented on
    other pages, so a re-export (e.g. chat's ChatCompletion, documented
    under types.md) is not duplicated onto every page that re-exports it.
    """
    lines = text.splitlines()
    entries = parse_entries(lines)

    covered = set(extra_covered)  # canonical (module, qualname) identities
    whole_modules = set()    # modules rendered wholesale via `::: pkg.mod`
    module_entries = {}      # module -> [line indices of its entries]
    stale = []
    for idx, target in entries:
        obj, status = resolve(target)
        if status == "missing_attr":
            stale.append((idx, target))
            continue
        if status == "import_error":
            IMPORT_WARNINGS.append((page, target))
            continue
        if inspect.ismodule(obj):
            whole_modules.add(target)
        else:
            covered.add((obj.__module__, obj.__qualname__))
        module_entries.setdefault(entry_module(target, obj), []).append(idx)

    # Discovery scope: referenced modules + owned public subtree modules.
    scope = set() if page in FROZEN_PAGES else set(module_entries)
    for prefix in PAGE_SUBTREES[page]:
        for name in walk_subtree(prefix):
            if owner_page(name) == page:
                scope.add(name)

    additions = {}   # module -> [new symbol names]
    for name in sorted(scope):
        if name in whole_modules or name in EXCLUDE_MODULES:
            continue
        if any(name == p or name.startswith(p + ".") for p in EXCLUDE_PREFIXES):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf.startswith("_") and name not in module_entries:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            IMPORT_WARNINGS.append((page, f"{name} ({exc})"))
            continue
        exported = getattr(module, "__all__", None)
        if exported is not None:
            # Defined here AND explicitly exported: re-export surfaces
            # (agentstudio root, RL compatibility layer) stay curated.
            export_set = set(exported)
            candidates = [
                (symbol, getattr(module, symbol))
                for symbol in public_defined(module)
                if symbol in export_set
            ]
        elif name in module_entries:
            # Referenced but no __all__: only grow with defined classes.
            candidates = [
                (symbol, getattr(module, symbol))
                for symbol in public_defined(module, include_functions=False)
            ]
        else:
            continue
        for symbol, obj in candidates:
            key = (obj.__module__, obj.__qualname__)
            if key not in covered:
                covered.add(key)
                additions.setdefault(name, []).append(symbol)

    # Placement: after the module's last existing entry, or a new section.
    anchors = {}     # line index -> [directive lines to insert after it]
    new_sections = []
    for module, symbols in additions.items():
        directives = [f"::: {module}.{s}" for s in symbols]
        if module in module_entries:
            anchor = module_entries[module][-1]
            anchors.setdefault(anchor, []).extend(directives)
        else:
            title = module.rsplit(".", 1)[-1].replace("_", " ").title()
            new_sections.append((title, directives))

    stale_idx = {idx for idx, _ in stale}
    out = []
    for i, line in enumerate(lines):
        if i not in stale_idx:
            out.append(line)
        if i in anchors:
            for directive in anchors[i]:
                out.extend(["", directive])
    for title, directives in new_sections:
        if out and out[-1] != "":
            out.append("")
        out.extend([f"## {title}", ""])
        for j, directive in enumerate(directives):
            if j:
                out.append("")
            out.append(directive)
    new_text = "\n".join(out) + "\n"
    added = [d for ds in anchors.values() for d in ds]
    added += [d for _, ds in new_sections for d in ds]
    return new_text, added, [t for _, t in stale]


def collect_targets(text: str):
    return [target for _, target in parse_entries(text.splitlines())]


def find_unclassified(referenced):
    """Public modules neither documented nor declared internal.

    These are the blind spot of the publication contract: without this
    check a new module lacking __all__ would simply never be discovered.
    Modules that fail to import are skipped (optional dependency).
    """
    gaps = []
    for page, subtrees in PAGE_SUBTREES.items():
        for prefix in subtrees:
            for name in walk_subtree(prefix):
                if owner_page(name) != page or name in referenced:
                    continue
                if name in EXCLUDE_MODULES or _is_internal(name):
                    continue
                if any(name == p or name.startswith(p + ".")
                       for p in EXCLUDE_PREFIXES):
                    continue
                if name.rsplit(".", 1)[-1].startswith("_"):
                    continue
                try:
                    module = importlib.import_module(name)
                except Exception:
                    continue
                if not hasattr(module, "__all__"):
                    gaps.append(name)
    return sorted(set(gaps))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail instead of writing when pages drift")
    args = parser.parse_args(argv)

    drift = False
    locale_targets = {}
    IMPORT_WARNINGS.clear()
    # Canonical identities documented anywhere in the reference (en is
    # the source of truth; zh mirrors it).
    global_covered = set()
    global_referenced = set()
    for page in PAGE_SUBTREES:
        text = (REFERENCE_DIRS["en"] / page).read_text(encoding="utf-8")
        for _, target in parse_entries(text.splitlines()):
            obj, status = resolve(target)
            if status == "ok":
                global_referenced.add(entry_module(target, obj))
                if not inspect.ismodule(obj):
                    global_covered.add((obj.__module__, obj.__qualname__))
    for locale, directory in REFERENCE_DIRS.items():
        locale_targets[locale] = {}
        for page in PAGE_SUBTREES:
            path = directory / page
            text = path.read_text(encoding="utf-8")
            new_text, added, removed = sync_page(text, page, global_covered)
            locale_targets[locale][page] = collect_targets(new_text)
            if new_text != text:
                drift = True
                print(f"{locale}/{page}:")
                for directive in added:
                    print(f"  + {directive}")
                for target in removed:
                    print(f"  - {target} (no longer resolves)")
                if not args.check:
                    path.write_text(new_text, encoding="utf-8")

    # The two locales must document the same symbols.
    for page in PAGE_SUBTREES:
        en = locale_targets["en"][page]
        zh = locale_targets["zh"][page]
        if sorted(en) != sorted(zh):
            drift = True
            print(f"{page}: en/zh entry sets differ: "
                  f"{sorted(set(en) ^ set(zh))}", file=sys.stderr)

    # Blind-spot guard: every public module must be published (__all__
    # or a page entry) or explicitly declared internal.
    gaps = find_unclassified(global_referenced)
    if gaps:
        print("Unclassified public modules (neither documented nor "
              "declared internal):", file=sys.stderr)
        for name in gaps:
            print(f"  ? {name}", file=sys.stderr)
        print("Fix: add __all__ to publish the module, or list it in "
              "INTERNAL_MODULES / INTERNAL_PREFIXES in "
              "tools/gen_reference.py.", file=sys.stderr)
        if args.check:
            drift = True

    if IMPORT_WARNINGS:
        print("Kept despite import failure (optional dependency missing "
              "or module renamed — verify manually):", file=sys.stderr)
        for page, target in sorted(set(IMPORT_WARNINGS)):
            print(f"  ! {page}: {target}", file=sys.stderr)

    if args.check and drift:
        print("Reference pages are out of sync with the public API. "
              "Run: python tools/gen_reference.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
