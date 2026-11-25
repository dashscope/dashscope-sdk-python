# -*- coding: utf-8 -*-
"""CI guard: the MkDocs API reference must cover the package's public API.

Runs tools/gen_reference.py in --check mode: every ``:::`` directive
must resolve, both locales must document the same symbols, and no public
API symbol may be missing from the pages. A failure prints the exact
directives to add — fix by running ``python tools/gen_reference.py``.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "tools" / "gen_reference.py"

spec = importlib.util.spec_from_file_location("gen_reference", SCRIPT_PATH)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def _page_text(locale, page):
    return (gen.REFERENCE_DIRS[locale] / page).read_text(encoding="utf-8")


def test_every_directive_resolves():
    for locale in gen.REFERENCE_DIRS:
        for page in gen.PAGE_SUBTREES:
            for _, target in gen.parse_entries(
                _page_text(locale, page).splitlines(),
            ):
                _, status = gen.resolve(target)
                assert status == "ok", f"{locale}/{page}: {target} ({status})"


def test_reference_pages_in_sync_with_public_api():
    assert gen.main(["--check"]) == 0, (
        "Reference pages drifted from the public API. "
        "Run: python tools/gen_reference.py"
    )


def test_locales_document_same_symbols():
    for page in gen.PAGE_SUBTREES:
        en = gen.collect_targets(_page_text("en", page))
        zh = gen.collect_targets(_page_text("zh", page))
        assert sorted(en) == sorted(zh), f"{page}: en/zh entry sets differ"


def test_new_public_class_is_discovered(monkeypatch):
    """A class added to a referenced module (and its __all__) is picked up."""
    embeddings = importlib.import_module("dashscope.resources.embeddings")
    fake = type(
        "FakeNewModel",
        (),
        {"__module__": "dashscope.resources.embeddings"},
    )
    monkeypatch.setitem(vars(embeddings), "FakeNewModel", fake)
    monkeypatch.setattr(
        embeddings,
        "__all__",
        list(embeddings.__all__) + ["FakeNewModel"],
    )
    text = _page_text("en", "resources.md")
    new_text, added, removed = gen.sync_page(text, "resources.md")
    directive = "::: dashscope.resources.embeddings.FakeNewModel"
    assert directive in added
    assert not removed
    # inserted right after the module's last existing entry, prose intact
    anchor = "::: dashscope.resources.embeddings.EmbeddingResponse"
    assert new_text.index(anchor) < new_text.index(directive)
    assert "## TingWu" in new_text


def test_new_module_with_all_gets_own_section(monkeypatch):
    name = "dashscope.resources.fake_new"
    module = types.ModuleType(name)
    module.__all__ = ["Widget"]
    module.Widget = type("Widget", (), {"__module__": name})
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        gen,
        "walk_subtree",
        lambda prefix: [prefix, name],
    )
    new_text, added, _ = gen.sync_page(
        _page_text("en", "resources.md"),
        "resources.md",
    )
    assert "::: dashscope.resources.fake_new.Widget" in added
    assert "## Fake New" in new_text


def test_unexported_class_is_not_discovered(monkeypatch):
    """Without __all__ membership a new class stays undocumented."""
    embeddings = importlib.import_module("dashscope.resources.embeddings")
    fake = type(
        "InternalHelper",
        (),
        {"__module__": "dashscope.resources.embeddings"},
    )
    monkeypatch.setitem(vars(embeddings), "InternalHelper", fake)
    _, added, _ = gen.sync_page(
        _page_text("en", "resources.md"),
        "resources.md",
    )
    assert "::: dashscope.resources.embeddings.InternalHelper" not in added


def test_stale_entry_is_removed():
    text = (
        "# Page\n\n## Section\n\n"
        "::: dashscope.resources.embeddings.Embeddings\n\n"
        "::: dashscope.resources.embeddings.NoSuchSymbol\n"
    )
    new_text, _, removed = gen.sync_page(text, "resources.md")
    assert removed == ["dashscope.resources.embeddings.NoSuchSymbol"]
    assert "NoSuchSymbol" not in new_text
    assert "::: dashscope.resources.embeddings.Embeddings" in new_text


def test_legacy_page_is_frozen(monkeypatch):
    """legacy_1x entries are checked but the page never grows."""
    generation = importlib.import_module(
        "dashscope.legacy_1x.aigc.generation",
    )
    fake = type(
        "LegacyNewThing",
        (),
        {"__module__": "dashscope.legacy_1x.aigc.generation"},
    )
    monkeypatch.setitem(vars(generation), "LegacyNewThing", fake)
    _, added, _ = gen.sync_page(_page_text("en", "legacy.md"), "legacy.md")
    assert added == []


def _fake_public_module(monkeypatch, name, with_all):
    module = types.ModuleType(name)
    module.Widget = type("Widget", (), {"__module__": name})
    if with_all:
        module.__all__ = ["Widget"]
    monkeypatch.setitem(sys.modules, name, module)
    original_walk = gen.walk_subtree

    def patched_walk(prefix):
        names = original_walk(prefix)
        if prefix == "dashscope.resources":
            names = names + [name]
        return names

    monkeypatch.setattr(gen, "walk_subtree", patched_walk)


def test_unclassified_public_module_fails_check(monkeypatch):
    """Blind-spot guard: a new public module without __all__ or an
    internal classification must fail --check, not slip through."""
    _fake_public_module(
        monkeypatch,
        "dashscope.resources.sneaky_new",
        with_all=False,
    )
    assert gen.main(["--check"]) == 1


def test_declaring_module_internal_passes_check(monkeypatch):
    name = "dashscope.resources.sneaky_new"
    _fake_public_module(monkeypatch, name, with_all=False)
    monkeypatch.setattr(gen, "INTERNAL_MODULES", gen.INTERNAL_MODULES | {name})
    assert gen.main(["--check"]) == 0


def test_publishing_module_with_all_is_discovered(monkeypatch):
    """With __all__ the module is published: --check reports the missing
    page entries (drift) rather than an unclassified gap."""
    _fake_public_module(
        monkeypatch,
        "dashscope.resources.sneaky_new",
        with_all=True,
    )
    assert gen.main(["--check"]) == 1  # pages need the new entries
    _, added, _ = gen.sync_page(
        _page_text("en", "resources.md"),
        "resources.md",
    )
    assert "::: dashscope.resources.sneaky_new.Widget" in added


def test_import_warnings_are_collected(monkeypatch):
    gen.IMPORT_WARNINGS.clear()
    # No importable prefix at all -> import_error (kept + reported),
    # as opposed to a missing attribute on an importable module (stale).
    text = "# Page\n\n::: totally_missing_pkg.Thing\n"
    gen.sync_page(text, "resources.md")
    assert any(
        "totally_missing_pkg.Thing" in target
        for _, target in gen.IMPORT_WARNINGS
    )
    gen.IMPORT_WARNINGS.clear()
