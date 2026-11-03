# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sync/async twin parity checks (T7 phase 1).

The sync and async classes in ``dashscope`` are maintained by hand as
parallel copies. Nothing flags it when a method or a parameter is added to
one side and forgotten on the other -- the drift only surfaces at runtime,
usually as an ``AttributeError`` on the async client. These checks parse
each twin pair with :mod:`ast` and assert the two sides expose the same
methods with the same parameters, so drift fails CI.

Three things are compared: the method-name set, each method's parameter
names/kinds, and the *body* of every method whose two halves are mechanical
copies of each other. Docstrings, annotations and defaults are ignored
throughout -- the async twin legitimately differs there (a one-line "Async
twin of ..." docstring, a ``Callable[..., Awaitable[None]]`` annotation).
Comparing them would report false positives, not real drift.

The body check normalizes away exactly the sync/async spelling difference
(``async def`` / ``await``, the ``Async`` class-name prefix, ``aclose`` and
the four async dunders) and compares :func:`ast.unparse` output rather than
source lines, so it is immune to re-formatting: dropping an ``await``
shortens a line and the formatter may then re-flow it. Most shared public
methods are identical under that normalization and are checked. The rest
diverge structurally -- a sync ``yield`` generator against an async ``async
for`` one, ``time.sleep`` against ``asyncio.sleep``, ``run_blocking``
offloading a blocking upload to a worker thread -- and are listed in
``_BODY_EXEMPT`` with the reason. That list is *self-expiring*:
``test_declared_exemptions_are_still_needed`` fails once an exemption stops
describing a real difference, so fixing a gap retires its own blind spot
instead of leaving the method unchecked forever.

Two further exclusions keep the signal honest:

* **Underscore-prefixed helpers are not compared.** A private method is not
  a contract, and the two sides routinely reach the same behaviour by
  different private means (``Stream._iterate`` is a sync generator; the
  async side inlines an async generator in ``__aiter__``). Dunder methods
  *are* compared -- they are protocol surface, not helpers.
* **Declared asymmetries are listed per pair.** A websocket-only method with
  no async counterpart is a design fact, not drift; naming it in the pair's
  ``sync_only`` set documents the decision and keeps the check red for
  anything *not* declared.

``test_every_async_twin_in_the_v2_tree_is_covered`` closes the loop: it
re-discovers the twins from the source tree and fails if a new resource
ships an async class that nobody added to ``_PAIRS``, so coverage cannot rot
silently.

Why detection rather than generation: T7's second phase floated building the
async side from the sync one with ``unasync``. Measured against this tree the
trade is poor -- about a third of the shared methods diverge structurally (a
sync ``yield`` generator against an ``async for``, ``time.sleep`` against
``asyncio.sleep``, ``asyncio.run()`` against a bare ``await``,
``run_blocking`` offloading a blocking upload to a worker thread), so
generation would cover part of the tree and leave the rest hand-written
beside it. It would also swap ``_PAIRS`` for a table of rewrite rules and
per-file overrides -- the same hand-maintained list in a different shape --
while making the async files read-only and pointing stack traces at
generated code. ``_BODY_EXEMPT`` below is the honest record of where the two
halves really are not copies.

Out of scope: ``legacy_1x`` (archived) and ``applications/agentstudio`` (a
separate sub-SDK with its own pagination and streaming twins).
"""

from __future__ import annotations

import ast
import copy
import difflib
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories whose twins are deliberately not checked; see the docstring.
_SKIP_DIRS = frozenset({"legacy_1x", "applications"})

# Twins whose two halves do not share a stem, so the ``Async``-strip rule in
# ``_discover_pairs`` cannot pair them. Async name -> sync name.
_RENAMED_TWINS = {"AsyncSpinner": "StderrSpinner"}

# An async-side method whose sync counterpart carries a different name.
# Keys are the async spelling, values the sync one; both sides are compared
# under the mapped name, so the exemptions below use the sync spelling too.
# Everything else is spelled the same on both sides (``Jobs.wait``, every
# resource method), so no other mapping is needed.
_NAME_MAP = {
    "aclose": "close",
    "__aenter__": "__enter__",
    "__aexit__": "__exit__",
    "__aiter__": "__iter__",
    "__anext__": "__next__",
}


class _Pair(NamedTuple):
    """One sync/async twin to compare, plus its declared asymmetries.

    ``sync_only`` / ``async_only`` exempt a *method name* that intentionally
    exists on one side alone. ``params_exempt`` exempts a method present on
    both sides whose signature intentionally differs. Every use of these
    fields carries a comment saying why, because an unexplained exemption is
    how real drift gets hidden.
    """

    path: str
    sync_cls: str
    async_cls: str
    sync_only: FrozenSet[str] = frozenset()
    async_only: FrozenSet[str] = frozenset()
    params_exempt: FrozenSet[str] = frozenset()


_PAIRS: List[_Pair] = [
    # ``ws_connection`` is the pure-sync websocket helper (thread model);
    # AsyncClient reaches the same sockets through ``connector=`` arguments.
    _Pair(
        "dashscope/client.py",
        "Client",
        "AsyncClient",
        sync_only=frozenset({"ws_connection"}),
    ),
    _Pair("dashscope/transport/http.py", "SyncTransport", "AsyncTransport"),
    _Pair("dashscope/transport/sse.py", "Stream", "AsyncStream"),
    _Pair("dashscope/resources/_jobs.py", "Jobs", "AsyncJobs"),
    _Pair("dashscope/resources/apps.py", "Apps", "AsyncApps"),
    _Pair(
        "dashscope/resources/apps.py",
        "AppCompletions",
        "AsyncAppCompletions",
    ),
    _Pair("dashscope/resources/apps.py", "AppStream", "AsyncAppStream"),
    _Pair("dashscope/resources/audio/__init__.py", "Audio", "AsyncAudio"),
    _Pair(
        "dashscope/resources/audio/asr.py",
        "TranscriptionJobs",
        "AsyncTranscriptionJobs",
    ),
    # ``recognize`` / ``recognize_file`` drive a microphone or a file through
    # a websocket (they take ``ws_url`` / ``connector``), which is the sync
    # thread model with no async counterpart -- the same reason
    # ``transport/ws.py`` has no twin at all.
    _Pair(
        "dashscope/resources/audio/asr.py",
        "Asr",
        "AsyncAsr",
        sync_only=frozenset({"recognize", "recognize_file"}),
    ),
    _Pair(
        "dashscope/resources/audio/http_tts.py",
        "HttpSynthesisStream",
        "AsyncHttpSynthesisStream",
        # ``__aiter__`` hands back the inner ``AsyncIterator``, so there is
        # no ``__anext__`` to delegate; the sync side implements both itself.
        sync_only=frozenset({"__next__"}),
    ),
    # ``AsyncHttpTts`` takes ``(text, **kwargs)`` and pops the same keys the
    # sync side declares explicitly, so behaviour matches but the signature
    # does not. Exempted rather than left silently green: tightening the
    # async signature to the 12 typed kwargs is tracked separately, and this
    # entry is what keeps that gap visible.
    _Pair(
        "dashscope/resources/audio/http_tts.py",
        "HttpTts",
        "AsyncHttpTts",
        params_exempt=frozenset({"stream", "synthesize"}),
    ),
    _Pair("dashscope/resources/chat/__init__.py", "Chat", "AsyncChat"),
    _Pair(
        "dashscope/resources/chat/completions.py",
        "Completions",
        "AsyncCompletions",
    ),
    _Pair(
        "dashscope/resources/chat/completions.py",
        "ChatStream",
        "AsyncChatStream",
    ),
    _Pair("dashscope/resources/decision.py", "Decision", "AsyncDecision"),
    _Pair(
        "dashscope/resources/deployments.py",
        "Deployments",
        "AsyncDeployments",
    ),
    _Pair(
        "dashscope/resources/embeddings.py",
        "EmbeddingJobs",
        "AsyncEmbeddingJobs",
    ),
    _Pair(
        "dashscope/resources/embeddings.py",
        "Embeddings",
        "AsyncEmbeddings",
    ),
    _Pair("dashscope/resources/files.py", "Files", "AsyncFiles"),
    _Pair("dashscope/resources/fine_tunes.py", "FineTunes", "AsyncFineTunes"),
    _Pair("dashscope/resources/images.py", "ImageJobs", "AsyncImageJobs"),
    _Pair("dashscope/resources/images.py", "Images", "AsyncImages"),
    _Pair("dashscope/resources/models.py", "Models", "AsyncModels"),
    _Pair(
        "dashscope/resources/multimodal/__init__.py",
        "Multimodal",
        "AsyncMultimodal",
    ),
    _Pair(
        "dashscope/resources/multimodal/chat.py",
        "MultimodalChat",
        "AsyncMultimodalChat",
    ),
    _Pair(
        "dashscope/resources/multimodal/embeddings.py",
        "MultimodalEmbeddings",
        "AsyncMultimodalEmbeddings",
    ),
    _Pair(
        "dashscope/resources/reinforcement_learning.py",
        "RlJobs",
        "AsyncRlJobs",
    ),
    _Pair("dashscope/resources/reinforcement_learning.py", "Rl", "AsyncRl"),
    _Pair("dashscope/resources/rerank.py", "Rerank", "AsyncRerank"),
    # ``realtime`` opens a websocket session (``ws_url`` / ``connector`` /
    # ``connection``), so it stays sync-only like the other ws entry points.
    _Pair(
        "dashscope/resources/tingwu.py",
        "TingWu",
        "AsyncTingWu",
        sync_only=frozenset({"realtime"}),
    ),
    _Pair(
        "dashscope/resources/tokenization.py",
        "Tokenization",
        "AsyncTokenization",
    ),
    _Pair("dashscope/resources/videos.py", "VideoJobs", "AsyncVideoJobs"),
    _Pair("dashscope/resources/videos.py", "Videos", "AsyncVideos"),
    # A CLI progress spinner, copied by hand like everything else here. Its
    # contract is the context-manager protocol, so the dunder comparison
    # (mapped through ``_NAME_MAP``) is the whole check.
    _Pair(
        "dashscope/acli/utils/spinner.py",
        "StderrSpinner",
        "AsyncSpinner",
    ),
]

_IDS = [pair.sync_cls for pair in _PAIRS]

# Methods whose two halves are deliberately *not* mechanical copies, so their
# bodies are not compared. Each value states the reason, because an
# unexplained exemption is how real drift hides.
#
# These are self-expiring: ``test_declared_exemptions_are_still_needed``
# fails the moment an entry stops describing a real difference, so closing
# one of these gaps retires its own exemption instead of leaving the method
# permanently unchecked. A key that names no real method fails too, which is
# what keeps a rename from silently orphaning an entry.
#
# Keyed by ``(path, sync_cls, method)``, the method in its sync spelling --
# so ``aclose`` appears as ``close`` and ``__aexit__`` as ``__exit__``.
_BodyKey = Tuple[str, str, str]

_BODY_EXEMPT: Dict[_BodyKey, str] = {
    # -- sync generator against an async generator -----------------------
    # ``yield`` versus ``async for`` is a different construct, not a token
    # rewrite. This is why T7 scopes streaming out of the generation pilot.
    ("dashscope/transport/sse.py", "Stream", "__iter__"): (
        "sync is a generator over EventSource with a try/finally close; "
        "async returns the _iterate() async generator"
    ),
    ("dashscope/resources/apps.py", "AppStream", "__iter__"): (
        "iter(self._inner) against self._inner.__iter__(), the async "
        "iterator protocol having no iter() builtin equivalent"
    ),
    ("dashscope/resources/apps.py", "AppStream", "text_stream"): (
        "sync yields in a for loop; async returns the _text_stream() "
        "async generator"
    ),
    (
        "dashscope/resources/audio/http_tts.py",
        "HttpSynthesisStream",
        "__iter__",
    ): (
        "sync returns self and implements __next__; async returns "
        "self._inner, whose __anext__ does the work"
    ),
    (
        "dashscope/resources/audio/http_tts.py",
        "HttpSynthesisStream",
        "close",
    ): (
        "sync closes the inner stream then the response in a try/finally; "
        "async has no inner sync stream to close"
    ),
    (
        "dashscope/resources/chat/completions.py",
        "ChatStream",
        "__iter__",
    ): (
        "sync accumulates over iter(self._inner); async over the "
        "_accumulated() async generator"
    ),
    (
        "dashscope/resources/chat/completions.py",
        "ChatStream",
        "text_stream",
    ): "sync yields in a for loop; async returns _text_stream()",
    (
        "dashscope/resources/chat/completions.py",
        "ChatStream",
        "reasoning_stream",
    ): "sync yields in a for loop; async returns _reasoning_stream()",
    (
        "dashscope/resources/chat/completions.py",
        "ChatStream",
        "tool_call_stream",
    ): "sync yields in a for loop; async returns _tool_call_stream()",
    (
        "dashscope/resources/chat/completions.py",
        "ChatStream",
        "get_final_message",
    ): (
        "async must materialize the chunks with an async comprehension "
        "before aggregating; sync passes the live iterator straight in"
    ),
    # -- SSE consumption: ``for`` against ``async for`` ------------------
    ("dashscope/transport/http.py", "SyncTransport", "open_stream"): (
        "async builds the request up front, reads it with aread() and "
        "sleeps with _asleep; T7 scopes open_stream out of the pilot"
    ),
    (
        "dashscope/resources/deployments.py",
        "Deployments",
        "stream_events",
    ): "for _iter_sse(response) against async for _aiter_sse(response)",
    (
        "dashscope/resources/fine_tunes.py",
        "FineTunes",
        "stream_events",
    ): "for _iter_sse(response) against async for _aiter_sse(response)",
    # -- blocking work offloaded to a worker thread ----------------------
    # The async twin cannot block the event loop, so file upload and
    # request building go through run_blocking and the send through _asend.
    ("dashscope/resources/images.py", "Images", "generate"): (
        "_upload_inputs runs inline; async offloads it via run_blocking"
    ),
    ("dashscope/resources/images.py", "ImageJobs", "create"): (
        "_upload_inputs runs inline; async offloads it via run_blocking"
    ),
    ("dashscope/resources/videos.py", "VideoJobs", "create"): (
        "_upload_inputs runs inline; async offloads it via run_blocking"
    ),
    (
        "dashscope/resources/multimodal/embeddings.py",
        "MultimodalEmbeddings",
        "create",
    ): (
        "upload_embedding_media runs inline; async offloads it via "
        "run_blocking"
    ),
    (
        "dashscope/resources/chat/completions.py",
        "Completions",
        "create",
    ): (
        "_build_request runs inline and sends via _send; async offloads "
        "the build via run_blocking and sends via _asend"
    ),
    (
        "dashscope/resources/multimodal/chat.py",
        "MultimodalChat",
        "create",
    ): (
        "_build_request runs inline and sends via _send; async offloads "
        "the build via run_blocking and sends via _asend"
    ),
    # -- retry backoff: a different sleep helper -------------------------
    ("dashscope/transport/http.py", "SyncTransport", "request"): (
        "backoff calls the _sleep helper; async calls its _asleep twin"
    ),
    # -- polling: time.sleep against asyncio.sleep -----------------------
    ("dashscope/resources/_jobs.py", "Jobs", "wait"): (
        "defaults the injectable _sleep to time.sleep; async to "
        "asyncio.sleep"
    ),
    ("dashscope/resources/audio/asr.py", "TranscriptionJobs", "wait"): (
        "defaults the injectable _sleep to time.sleep; async to "
        "asyncio.sleep"
    ),
    ("dashscope/resources/fine_tunes.py", "FineTunes", "wait"): (
        "defaults the injectable _sleep to time.sleep; async to "
        "asyncio.sleep"
    ),
    (
        "dashscope/resources/reinforcement_learning.py",
        "RlJobs",
        "wait",
    ): (
        "defaults the injectable _sleep to time.sleep; async to "
        "asyncio.sleep"
    ),
    # -- sync drives an async workflow through asyncio.run() -------------
    # Every ``Rl`` entry point shells into the async-only ``_agentic_rl``
    # client, so the sync side needs a loop and the async side already has
    # one.
    (
        "dashscope/resources/reinforcement_learning.py",
        "RlJobs",
        "create",
    ): "sync wraps rl.run() in asyncio.run(); async awaits it",
    (
        "dashscope/resources/reinforcement_learning.py",
        "Rl",
        "register_functions",
    ): (
        "sync wraps _register_functions_workflow in asyncio.run(); async "
        "awaits it"
    ),
    (
        "dashscope/resources/reinforcement_learning.py",
        "Rl",
        "test_functions",
    ): "sync wraps the _agentic_rl coroutine in asyncio.run()",
    (
        "dashscope/resources/reinforcement_learning.py",
        "Rl",
        "upload_data",
    ): "sync wraps rl.upload_datasets in asyncio.run()",
    (
        "dashscope/resources/reinforcement_learning.py",
        "Rl",
        "query_function_instance_logs",
    ): "sync wraps the _agentic_rl coroutine in asyncio.run()",
    (
        "dashscope/resources/reinforcement_learning.py",
        "Rl",
        "query_all_function_instance_logs",
    ): "sync wraps the _agentic_rl coroutine in asyncio.run()",
    (
        "dashscope/resources/reinforcement_learning.py",
        "Rl",
        "delete_function_instance",
    ): "sync wraps the _agentic_rl coroutine in asyncio.run()",
    # -- constructors wiring a different object graph ----------------------
    ("dashscope/client.py", "Client", "__init__"): (
        "builds a SyncTransport; the async client builds an AsyncTransport "
        "imported under a local alias"
    ),
    ("dashscope/client.py", "Client", "agentstudio"): (
        "instantiates the agentstudio sync Client; async instantiates that "
        "sub-SDK's own AsyncClient"
    ),
    ("dashscope/resources/audio/__init__.py", "Audio", "__init__"): (
        "async wraps each sub-resource in _AsyncResourceProxy and has no "
        "``realtime`` (websocket, sync-only)"
    ),
    ("dashscope/resources/audio/asr.py", "Asr", "__init__"): (
        "sync binds self._client itself; async relies on the base wiring"
    ),
    # -- spinner: a thread against an asyncio.Task -----------------------
    ("dashscope/acli/utils/spinner.py", "StderrSpinner", "__init__"): (
        "sync initialises a _stop flag; async an _task handle"
    ),
    ("dashscope/acli/utils/spinner.py", "StderrSpinner", "__enter__"): (
        "sync starts a daemon threading.Thread; async schedules "
        "asyncio.create_task"
    ),
    ("dashscope/acli/utils/spinner.py", "StderrSpinner", "__exit__"): (
        "sync sets _stop and joins the thread; async cancels the task and "
        "swallows CancelledError"
    ),
}

# One parameter, normalized: its kind and its name.
_Param = Tuple[str, str]
_Sig = List[_Param]


def _is_private(name: str) -> bool:
    """Whether ``name`` is an internal helper, exempt from parity.

    Underscore-prefixed but *not* a dunder: ``__aiter__`` / ``__aenter__``
    are protocol surface the caller depends on, so they stay in scope and
    are compared under their ``_NAME_MAP`` sync spelling.
    """
    return name.startswith("_") and not name.startswith("__")


def _params(fn) -> _Sig:
    """Normalized ``(kind, name)`` parameters of ``fn``, ``self`` dropped.

    ``kind`` is one of ``pos`` / ``var`` / ``kw`` / ``kwvar``. Annotations
    and defaults are not captured: they legitimately differ between twins.
    """
    args = fn.args
    out: _Sig = [("pos", a.arg) for a in args.args if a.arg != "self"]
    if args.vararg is not None:
        out.append(("var", args.vararg.arg))
    out += [("kw", a.arg) for a in args.kwonlyargs]
    if args.kwarg is not None:
        out.append(("kwvar", args.kwarg.arg))
    return out


def _sigs_from_source(src: str, cls: str) -> Dict[str, _Sig]:
    """Public method signatures of ``cls`` in ``src``, keyed by mapped name.

    Accepting a source string (rather than only a file) lets the guard tests
    below feed the detector synthetic twins with a known drift. ``@overload``
    repeats a name and the last binding wins, which is safe because an
    overload set shares parameter *names* -- only annotations and defaults
    vary across the stubs, and neither is compared.
    """
    tree = ast.parse(src)
    sigs: Dict[str, _Sig] = {}
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for fn in node.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_private(fn.name):
                continue
            sigs[_NAME_MAP.get(fn.name, fn.name)] = _params(fn)
    return sigs


def _deasync(name: str) -> str:
    """``AsyncModels`` -> ``Models``; anything else is left alone.

    Requiring an uppercase letter after the prefix keeps names that merely
    begin with those five characters -- ``Asyncio``, say -- from being
    mangled into something that no longer resolves.
    """
    if name.startswith("Async") and name[5:6].isupper():
        return name[5:]
    return name


class _Normalize(ast.NodeTransformer):
    """Rewrite a method so both halves of a twin normalize to one tree.

    This undoes exactly the mechanical half of the hand copy: ``await`` is
    dropped, ``async def`` becomes ``def``, the ``Async`` class-name prefix
    and the ``_NAME_MAP`` spellings collapse to their sync names, and
    docstrings plus signature annotations are removed. All of those differ
    legitimately between twins (an "Async twin of ..." one-liner, a
    ``Callable[..., Awaitable[None]]``), so comparing them is noise.

    Local variable annotations are kept -- they read the same on both sides,
    so they are signal. ``AsyncFor`` and ``AsyncWith`` are deliberately not
    rewritten: a sync ``for`` against an ``async for`` is a real structural
    difference, which is what ``_BODY_EXEMPT`` exists to declare.
    """

    def visit_Await(self, node):
        self.generic_visit(node)
        return node.value

    def visit_AsyncFunctionDef(self, node):
        plain = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
        )
        return self.visit_FunctionDef(plain)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.name = _NAME_MAP.get(node.name, node.name)
        node.returns = None
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        for arg in ast.walk(node.args):
            if isinstance(arg, ast.arg):
                arg.annotation = None
        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        node.id = _deasync(node.id)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr = _deasync(node.attr)
        node.attr = _NAME_MAP.get(attr, attr)
        return node


def _bodies_from_source(src: str, cls: str) -> Dict[str, str]:
    """Normalized body of each public method of ``cls``, keyed by name.

    ``@overload`` repeats a name and the last binding wins, so the
    implementation is what gets compared, on both sides alike.

    Comparing unparsed trees rather than source lines keeps the check honest
    under re-formatting: dropping an ``await`` shortens a line, and the
    formatter is then free to re-flow it, which a line diff would report as
    drift.
    """
    tree = ast.parse(src)
    bodies: Dict[str, str] = {}
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for fn in node.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_private(fn.name):
                continue
            clone = _Normalize().visit(copy.deepcopy(fn))
            clone.name = _NAME_MAP.get(clone.name, clone.name)
            ast.fix_missing_locations(clone)
            key = _NAME_MAP.get(fn.name, fn.name)
            bodies[key] = ast.unparse(clone)
    return bodies


def _diff(
    pair: _Pair,
    sync_sigs: Dict[str, _Sig],
    async_sigs: Dict[str, _Sig],
) -> Tuple[List[str], List[str]]:
    """Compare one twin pair; return ``(method_drift, param_drift)`` messages.

    Both lists empty means the pair aligns. Messages name the offending
    method (and, for parameters, both sides' signatures) so a failure points
    straight at the drift instead of at this file.
    """
    method_problems: List[str] = []
    param_problems: List[str] = []
    sync_names = set(sync_sigs) - pair.sync_only
    async_names = set(async_sigs) - pair.async_only
    for name in sorted(sync_names - async_names):
        method_problems.append(f"{pair.async_cls} is missing method: {name}")
    for name in sorted(async_names - sync_names):
        method_problems.append(f"{pair.sync_cls} is missing method: {name}")
    shared = (sync_names & async_names) - pair.params_exempt
    for name in sorted(shared):
        if sync_sigs[name] != async_sigs[name]:
            param_problems.append(
                f"{name} parameter drift: "
                f"{pair.sync_cls}={sync_sigs[name]} "
                f"{pair.async_cls}={async_sigs[name]}",
            )
    return method_problems, param_problems


def _diff_pair(pair: _Pair) -> Tuple[List[str], List[str]]:
    """Read both sides of ``pair`` off disk and diff them."""
    src = (REPO_ROOT / pair.path).read_text(encoding="utf-8")
    sync_sigs = _sigs_from_source(src, pair.sync_cls)
    async_sigs = _sigs_from_source(src, pair.async_cls)
    # Guard against a typo'd or renamed class silently comparing nothing,
    # which would make the whole pair vacuously green.
    assert sync_sigs, f"no methods found for {pair.sync_cls} in {pair.path}"
    assert async_sigs, f"no methods found for {pair.async_cls} in {pair.path}"
    return _diff(pair, sync_sigs, async_sigs)


def _bodies_of_pair(pair: _Pair) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Both sides' normalized method bodies, read off disk."""
    src = (REPO_ROOT / pair.path).read_text(encoding="utf-8")
    found: Dict[str, Dict[str, str]] = {}
    for cls in (pair.sync_cls, pair.async_cls):
        bodies = _bodies_from_source(src, cls)
        # Guard against a typo'd or renamed class silently comparing
        # nothing, which would make the whole pair vacuously green.
        assert bodies, f"no methods found for {cls} in {pair.path}"
        found[cls] = bodies
    return found[pair.sync_cls], found[pair.async_cls]


def _body_diff(
    pair: _Pair,
    sync_bodies: Dict[str, str],
    async_bodies: Dict[str, str],
    exempt: Optional[Dict[_BodyKey, str]] = None,
) -> List[str]:
    """Methods present on both sides whose normalized bodies drifted.

    Messages carry a unified diff of the normalized trees, so a failure
    shows the actual divergence rather than just naming the method.
    """
    if exempt is None:
        exempt = _BODY_EXEMPT
    problems: List[str] = []
    shared = (set(sync_bodies) - pair.sync_only) & (
        set(async_bodies) - pair.async_only
    )
    # A method whose signature already differs is reported by the parameter
    # check; re-reporting its body would bury that in a much longer diff.
    shared -= pair.params_exempt
    for name in sorted(shared):
        if (pair.path, pair.sync_cls, name) in exempt:
            continue
        if sync_bodies[name] == async_bodies[name]:
            continue
        diff = "\n".join(
            difflib.unified_diff(
                sync_bodies[name].splitlines(),
                async_bodies[name].splitlines(),
                pair.sync_cls,
                pair.async_cls,
                lineterm="",
                n=1,
            ),
        )
        problems.append(f"{name} body drifted:\n{diff}")
    return problems


def _body_diff_pair(pair: _Pair) -> List[str]:
    """Read both sides of ``pair`` off disk and diff their bodies."""
    sync_bodies, async_bodies = _bodies_of_pair(pair)
    return _body_diff(pair, sync_bodies, async_bodies)


def _discover_pairs() -> Set[Tuple[str, str, str]]:
    """Every ``(path, sync_cls, async_cls)`` twin in the checked tree.

    The rule is a shared module holding ``Foo`` and ``AsyncFoo``. Two
    exceptions keep it from silently missing a twin: ``SyncTransport``, whose
    sync side is prefixed rather than bare, so the ``Async``-strip is retried
    with a ``Sync`` prefix; and the pairs in ``_RENAMED_TWINS``, where the
    two halves do not share a stem at all.
    """
    found: Set[Tuple[str, str, str]] = set()
    for path in sorted((REPO_ROOT / "dashscope").rglob("*.py")):
        if _SKIP_DIRS.intersection(path.parts):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        for name in names:
            if not name.startswith("Async"):
                continue
            sync = _RENAMED_TWINS.get(name) or name[len("Async") :]
            if sync not in names and f"Sync{sync}" in names:
                sync = f"Sync{sync}"
            if sync in names:
                found.add((rel, sync, name))
    return found


@pytest.mark.parametrize("pair", _PAIRS, ids=_IDS)
def test_twin_method_sets_match(pair):
    method_problems, _ = _diff_pair(pair)
    detail = "\n".join(method_problems)
    msg = (
        f"{pair.path}: {pair.sync_cls}/{pair.async_cls} method set "
        f"drifted:\n{detail}"
    )
    assert method_problems == [], msg


@pytest.mark.parametrize("pair", _PAIRS, ids=_IDS)
def test_twin_param_lists_match(pair):
    _, param_problems = _diff_pair(pair)
    detail = "\n".join(param_problems)
    msg = (
        f"{pair.path}: {pair.sync_cls}/{pair.async_cls} parameters "
        f"drifted:\n{detail}"
    )
    assert param_problems == [], msg


@pytest.mark.parametrize("pair", _PAIRS, ids=_IDS)
def test_twin_method_bodies_match(pair):
    """A method that is a mechanical copy must stay a mechanical copy.

    The two checks above compare the contract only, so editing one side's
    *logic* and forgetting the other passes both. Most twins are byte-identical
    once the async spelling is normalized away, and those are held to it here;
    the structurally divergent ones are declared in ``_BODY_EXEMPT``.
    """
    problems = _body_diff_pair(pair)
    detail = "\n\n".join(problems)
    msg = (
        f"{pair.path}: {pair.sync_cls}/{pair.async_cls} method bodies "
        f"drifted:\n{detail}\n"
        "Mirror the change to both sides, or -- if the divergence is "
        "intentional -- add the method to _BODY_EXEMPT with the reason."
    )
    assert problems == [], msg


def test_every_async_twin_in_the_v2_tree_is_covered():
    """No twin may ship without a ``_PAIRS`` entry.

    Without this, adding ``AsyncWidgets`` to a new resource file escapes
    every check above -- the exact rot phase 1 exists to stop. The stale
    direction is asserted too, so a renamed or deleted class cannot leave a
    dead entry that quietly stops matching anything.
    """
    declared = {(p.path, p.sync_cls, p.async_cls) for p in _PAIRS}
    discovered = _discover_pairs()
    missing = sorted(discovered - declared)
    stale = sorted(declared - discovered)
    assert missing == [], (
        "sync/async twins with no parity entry -- add them to _PAIRS:\n"
        + "\n".join(f"  {m[0]}: {m[1]}/{m[2]}" for m in missing)
    )
    assert stale == [], (
        "_PAIRS entries that no longer match a twin in the tree (class "
        f"renamed or removed?): {stale}"
    )


def test_declared_exemptions_are_still_needed():
    """Every exemption must still describe a real difference.

    A stale exemption is worse than none: the method stops being compared
    and nothing says so, which is how a declared gap hardens into a
    permanent blind spot. Each kind is checked in the direction that retires
    it -- a ``sync_only`` method the async side has since grown, a
    ``params_exempt`` signature that has since been tightened, a
    ``_BODY_EXEMPT`` body that has since been made to match.

    An empty reason fails as well. The point of the list is that a reader
    can tell a design decision from an oversight.
    """
    stale: List[str] = []

    for pair in _PAIRS:
        src = (REPO_ROOT / pair.path).read_text(encoding="utf-8")
        sync_sigs = _sigs_from_source(src, pair.sync_cls)
        async_sigs = _sigs_from_source(src, pair.async_cls)
        for name in sorted(pair.sync_only):
            if name in async_sigs:
                stale.append(
                    f"{pair.path}: {pair.sync_cls}.{name} is exempt as "
                    f"sync-only, but {pair.async_cls} has it now too",
                )
        for name in sorted(pair.async_only):
            if name in sync_sigs:
                stale.append(
                    f"{pair.path}: {pair.async_cls}.{name} is exempt as "
                    f"async-only, but {pair.sync_cls} has it now too",
                )
        for name in sorted(pair.params_exempt):
            if name not in sync_sigs or name not in async_sigs:
                stale.append(
                    f"{pair.path}: {pair.sync_cls}.{name} is exempt from "
                    "the parameter check but is not on both sides",
                )
            elif sync_sigs[name] == async_sigs[name]:
                stale.append(
                    f"{pair.path}: {pair.sync_cls}.{name} parameters now "
                    "match -- drop it from params_exempt",
                )

    # _BODY_EXEMPT is keyed by (path, sync_cls), so each key must resolve to
    # exactly one twin; the length check is what makes that true.
    by_twin = {(p.path, p.sync_cls): p for p in _PAIRS}
    assert len(by_twin) == len(_PAIRS), (
        "_PAIRS holds a duplicate (path, sync_cls); _BODY_EXEMPT keys "
        "would be ambiguous"
    )

    cache: Dict[Tuple[str, str], Tuple[Dict[str, str], Dict[str, str]]] = {}
    for (path, sync_cls, name), reason in sorted(_BODY_EXEMPT.items()):
        where = f"{path}::{sync_cls}.{name}"
        pair = by_twin.get((path, sync_cls))
        if pair is None:
            stale.append(f"_BODY_EXEMPT {where} names no _PAIRS twin")
            continue
        if not reason.strip():
            stale.append(f"_BODY_EXEMPT {where} states no reason")
            continue
        if (path, sync_cls) not in cache:
            cache[(path, sync_cls)] = _bodies_of_pair(pair)
        sync_bodies, async_bodies = cache[(path, sync_cls)]
        if name not in sync_bodies or name not in async_bodies:
            stale.append(
                f"_BODY_EXEMPT {where} is not a public method on both "
                "sides (renamed?)",
            )
        elif sync_bodies[name] == async_bodies[name]:
            stale.append(
                f"{where} bodies now match -- drop it from _BODY_EXEMPT",
            )

    detail = "\n".join(f"  {s}" for s in stale)
    assert stale == [], (
        "exemptions that no longer describe a real difference:\n" + detail
    )


# ---------------------------------------------------------------------------
# Guard tests: prove the detector is not vacuous. These feed it synthetic
# twins with a known drift and assert it reports it -- the in-suite equivalent
# of "delete an async method and watch the parity test go red".
# ---------------------------------------------------------------------------

_SYNTHETIC = _Pair("synthetic.py", "Models", "AsyncModels")

_SYNC_SRC = """
class Models:
    def __init__(self, client):
        pass

    def list(self, *, page_no=1):
        pass

    def get(self, model):
        pass
"""

_ASYNC_SRC_ALIGNED = """
class AsyncModels:
    def __init__(self, client):
        pass

    async def list(self, *, page_no=1):
        pass

    async def get(self, model):
        pass
"""

_ASYNC_SRC_MISSING_METHOD = """
class AsyncModels:
    def __init__(self, client):
        pass

    async def list(self, *, page_no=1):
        pass
"""

_ASYNC_SRC_EXTRA_PARAM = """
class AsyncModels:
    def __init__(self, client):
        pass

    async def list(self, *, page_no=1, provider=None):
        pass

    async def get(self, model):
        pass
"""


def test_detector_accepts_an_aligned_twin():
    sync = _sigs_from_source(_SYNC_SRC, "Models")
    asyn = _sigs_from_source(_ASYNC_SRC_ALIGNED, "AsyncModels")
    method_problems, param_problems = _diff(_SYNTHETIC, sync, asyn)
    assert method_problems == []
    assert param_problems == []


def test_detector_flags_a_missing_async_method():
    sync = _sigs_from_source(_SYNC_SRC, "Models")
    asyn = _sigs_from_source(_ASYNC_SRC_MISSING_METHOD, "AsyncModels")
    method_problems, _ = _diff(_SYNTHETIC, sync, asyn)
    assert any("get" in problem for problem in method_problems)


def test_detector_flags_a_param_drift():
    sync = _sigs_from_source(_SYNC_SRC, "Models")
    asyn = _sigs_from_source(_ASYNC_SRC_EXTRA_PARAM, "AsyncModels")
    _, param_problems = _diff(_SYNTHETIC, sync, asyn)
    assert any("list" in problem for problem in param_problems)


def test_detector_refuses_to_compare_a_class_it_cannot_find():
    # A typo'd class name must blow up, not report a vacuous pass.
    pair = _Pair(
        "dashscope/resources/models.py",
        "Modelz",
        "AsyncModels",
    )
    with pytest.raises(AssertionError, match="no methods found"):
        _diff_pair(pair)


def test_name_map_pairs_async_spellings_with_their_sync_names():
    # close/aclose and the four async dunders are the same method under a
    # different name; the map must pair them, not report a two-sided drift.
    sync = _sigs_from_source(
        "class T:\n"
        "    def close(self):\n"
        "        pass\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *exc):\n"
        "        pass\n"
        "    def __iter__(self):\n"
        "        pass\n"
        "    def __next__(self):\n"
        "        pass\n",
        "T",
    )
    asyn = _sigs_from_source(
        "class T:\n"
        "    async def aclose(self):\n"
        "        pass\n"
        "    async def __aenter__(self):\n"
        "        pass\n"
        "    async def __aexit__(self, *exc):\n"
        "        pass\n"
        "    def __aiter__(self):\n"
        "        pass\n"
        "    async def __anext__(self):\n"
        "        pass\n",
        "T",
    )
    pair = _Pair("synthetic.py", "T", "T")
    method_problems, param_problems = _diff(pair, sync, asyn)
    assert method_problems == []
    assert param_problems == []


def test_sync_only_exemption_is_honored():
    # A method present only on the sync side is drift unless it is exempted;
    # the exemption (e.g. Client.ws_connection) must silence it.
    sync = _sigs_from_source(
        "class T:\n"
        "    def a(self):\n"
        "        pass\n"
        "    def b(self):\n"
        "        pass\n",
        "T",
    )
    asyn = _sigs_from_source(
        "class T:\n    async def a(self):\n        pass\n",
        "T",
    )
    declared = _Pair("synthetic.py", "T", "T")
    unexempted, _ = _diff(declared, sync, asyn)
    assert any("b" in problem for problem in unexempted)
    exempted, _ = _diff(
        _Pair("synthetic.py", "T", "T", sync_only=frozenset({"b"})),
        sync,
        asyn,
    )
    assert exempted == []


def test_private_helpers_are_not_compared():
    # A private helper is not a contract, and the twins legitimately reach
    # the same behaviour by different private means. Dunders stay compared.
    sync = _sigs_from_source(
        "class T:\n"
        "    def a(self):\n"
        "        pass\n"
        "    def _helper(self):\n"
        "        pass\n"
        "    def __iter__(self):\n"
        "        pass\n",
        "T",
    )
    asyn = _sigs_from_source(
        "class T:\n    async def a(self):\n        pass\n",
        "T",
    )
    assert "_helper" not in sync
    assert "__iter__" in sync
    method_problems, _ = _diff(_Pair("synthetic.py", "T", "T"), sync, asyn)
    # The dunder is still reported; the private helper is not.
    assert any("__iter__" in problem for problem in method_problems)
    assert not any("_helper" in problem for problem in method_problems)


def test_params_exempt_silences_a_declared_signature_gap():
    # The HttpTts/AsyncHttpTts case: both sides have the method, but the
    # async one takes ``**kwargs``. Exempting it must quiet the param check
    # while leaving the method-set check fully alive.
    sync = _sigs_from_source(
        "class T:\n    def s(self, text, *, model, voice):\n        pass\n",
        "T",
    )
    asyn = _sigs_from_source(
        "class T:\n    async def s(self, text, **kwargs):\n        pass\n",
        "T",
    )
    declared = _Pair("synthetic.py", "T", "T")
    _, unexempted = _diff(declared, sync, asyn)
    assert any("s" in problem for problem in unexempted)
    exempt, params = _diff(
        _Pair("synthetic.py", "T", "T", params_exempt=frozenset({"s"})),
        sync,
        asyn,
    )
    assert params == []
    assert exempt == []


# ---------------------------------------------------------------------------
# Guard tests for the body check. Same discipline as above: prove it catches
# real logic drift, and prove it does *not* fire on the differences the twins
# are supposed to have.
# ---------------------------------------------------------------------------


def test_body_check_ignores_the_mechanical_async_difference():
    # await, async def, the Async class prefix, a "twin of" docstring and a
    # signature annotation are all legitimate; none of them is drift.
    sync = _bodies_from_source(
        "class T:\n"
        "    def call(self, x):\n"
        "        helper = AsyncHelper()\n"
        "        return helper.run(x)\n",
        "T",
    )
    asyn = _bodies_from_source(
        "class T:\n"
        "    async def call(self, x: int) -> str:\n"
        '        """Async twin of T.call."""\n'
        "        helper = Helper()\n"
        "        return await helper.run(x)\n",
        "T",
    )
    pair = _Pair("synthetic.py", "T", "T")
    assert _body_diff(pair, sync, asyn) == []


def test_body_check_is_immune_to_reformatting():
    # Dropping an ``await`` shortens a line, so the formatter is free to
    # re-flow it. Comparing source lines would call that drift; comparing
    # unparsed trees does not.
    sync = _bodies_from_source(
        "class T:\n"
        "    def f(self, a, b):\n"
        "        return helper(a, b)\n",
        "T",
    )
    asyn = _bodies_from_source(
        "class T:\n"
        "    async def f(self, a, b):\n"
        "        return await helper(\n"
        "            a,\n"
        "            b,\n"
        "        )\n",
        "T",
    )
    pair = _Pair("synthetic.py", "T", "T")
    assert _body_diff(pair, sync, asyn) == []


def test_body_check_flags_a_logic_drift():
    # The contract checks above cannot see this: same method name, same
    # parameters, different behaviour on one side only.
    sync = _bodies_from_source(
        "class T:\n    def f(self, x):\n        return x + 1\n",
        "T",
    )
    asyn = _bodies_from_source(
        "class T:\n    async def f(self, x):\n        return x + 2\n",
        "T",
    )
    problems = _body_diff(_Pair("synthetic.py", "T", "T"), sync, asyn)
    assert len(problems) == 1
    assert "f body drifted" in problems[0]


def test_body_check_keeps_a_structural_async_difference():
    # ``async for`` is not a token rewrite, so it must still read as drift --
    # that is what forces the pair to declare itself in _BODY_EXEMPT rather
    # than diverge silently.
    sync = _bodies_from_source(
        "class T:\n"
        "    def f(self, xs):\n"
        "        for x in xs:\n"
        "            yield x\n",
        "T",
    )
    asyn = _bodies_from_source(
        "class T:\n"
        "    async def f(self, xs):\n"
        "        async for x in xs:\n"
        "            yield x\n",
        "T",
    )
    assert _body_diff(_Pair("synthetic.py", "T", "T"), sync, asyn) != []


def test_body_exempt_silences_a_declared_body_divergence():
    sync = _bodies_from_source(
        "class T:\n"
        "    def f(self, xs):\n"
        "        for x in xs:\n"
        "            yield x\n",
        "T",
    )
    asyn = _bodies_from_source(
        "class T:\n"
        "    async def f(self, xs):\n"
        "        async for x in xs:\n"
        "            yield x\n",
        "T",
    )
    pair = _Pair("synthetic.py", "T", "T")
    exempt = {("synthetic.py", "T", "f"): "sync generator vs async one"}
    assert _body_diff(pair, sync, asyn, exempt) == []
    # The exemption is per method: an undeclared one still reports.
    other = {("synthetic.py", "T", "g"): "unrelated"}
    assert _body_diff(pair, sync, asyn, other) != []


def test_body_check_compares_the_overload_implementation():
    # ``@overload`` repeats a name; the last binding is the implementation on
    # both sides, so that is what must be compared -- not a stub.
    sync = _bodies_from_source(
        "class T:\n"
        "    @overload\n"
        "    def f(self, x: int) -> int:\n"
        "        ...\n"
        "    def f(self, x):\n"
        "        return x * 2\n",
        "T",
    )
    asyn = _bodies_from_source(
        "class T:\n"
        "    @overload\n"
        "    async def f(self, x: int) -> int:\n"
        "        ...\n"
        "    async def f(self, x):\n"
        "        return await double(x)\n",
        "T",
    )
    assert "x * 2" in sync["f"]
    problems = _body_diff(_Pair("synthetic.py", "T", "T"), sync, asyn)
    assert len(problems) == 1
    assert "double" in problems[0]
