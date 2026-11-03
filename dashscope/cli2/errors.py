# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Exit-code tiers and error rendering for the dashscope CLI.

Every failure leaves the process with a code a script can branch on
without parsing stderr, and a one-line ``Error [category]: message`` on
stderr whose category is isomorphic to that code:

====  ==========  ==========================================
code  category    meaning
====  ==========  ==========================================
0     --          success
1     server      service-side failure, overload, or timeout
1     ratelimit   throttled (retryable, but not the caller's bug)
1     network     the request never completed
2     usage       the command line was wrong (matches click)
3     auth        bad credentials or insufficient permission
4     notfound    the named resource does not exist
====  ==========  ==========================================

Tier 2 deliberately matches the exit code click/Typer already use for
their own usage errors, so ``--help``-level mistakes and CLI-detected
parameter mistakes are indistinguishable to a caller -- which is the
point: both mean "fix your command line".

The server's own ``message`` is passed through verbatim. Re-wording it
would hide the detail a support ticket needs, so the CLI adds a category
label and nothing else.

Which category an exception lands in is *derived* from
``dashscope/_error_registry.py``, not hand-listed here. The catalog is
the standard: every ``PUBLIC_ERRORS.error_code`` names exactly one class
in ``dashscope.errors`` and carries that class's ``status_code``, so
``_STATUS_CATEGORY`` below is the CLI's one and only projection of the
catalog onto these tiers -- a row per status the catalog declares.
Regenerating the catalog therefore re-points the table with no edit in
this file; a *new status* is the one thing that needs a human decision,
and ``tests/cli2/test_errors.py`` fails until someone makes it.

SDK-side failures (``SdkError``) have no status in the catalog by
design, so they are categorised by their catalog ``code`` instead. The
catalog also ships ``solution`` text for them, which the CLI renders on
a separate dim line -- separate because it is the registry's wording,
not the service's, and splicing it into the message would blur which
words came from the gateway.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, NamedTuple, Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape

EXIT_OK = 0
EXIT_SERVER = 1
EXIT_USAGE = 2
EXIT_AUTH = 3
EXIT_NOT_FOUND = 4

USAGE = "usage"
AUTH = "auth"
RATELIMIT = "ratelimit"
SERVER = "server"
NETWORK = "network"
NOTFOUND = "notfound"

CATEGORIES = (USAGE, AUTH, RATELIMIT, SERVER, NETWORK, NOTFOUND)

_CATEGORY_EXIT = {
    USAGE: EXIT_USAGE,
    AUTH: EXIT_AUTH,
    RATELIMIT: EXIT_SERVER,
    SERVER: EXIT_SERVER,
    NETWORK: EXIT_SERVER,
    NOTFOUND: EXIT_NOT_FOUND,
}

# The CLI's projection of the catalog onto the tiers above: one row per
# status ``PUBLIC_ERRORS`` declares, keyed by status because that is the
# only thing the catalog states about a failure's nature. 401 and 403
# both mean "who are you / what may you do", 400 and 413 both mean "the
# request itself is wrong", and the three 5xx all mean "not your bug".
_STATUS_CATEGORY: Dict[int, str] = {
    400: USAGE,
    401: AUTH,
    403: AUTH,
    404: NOTFOUND,
    413: USAGE,
    429: RATELIMIT,
    500: SERVER,
    503: SERVER,
    504: SERVER,
}

# ``SDK_ERRORS`` entries carry no status, so a catalog ``code`` is the
# only handle. Deliberately short: an entry absent here is the SDK
# reporting something the CLI cannot attribute to the command line or to
# credentials, which is the ``server`` tier's meaning. ``sdk.IoError`` is
# the exception worth naming -- its own catalog solution reads "verify
# the file path exists and is readable", i.e. the caller's mistake.
_SDK_CODE_CATEGORY: Dict[str, str] = {
    "sdk.InvalidApiKey": AUTH,
    "sdk.ConnectionError": NETWORK,
    "sdk.Timeout": NETWORK,
    "sdk.IoError": USAGE,
}

# Environment kill switch for the deprecation notice the entry point
# prints before delegating to the 1.x command tree.
NO_DEPRECATION_ENV = "DASHSCOPE_CLI_NO_DEPRECATION"


class CliError(Exception):
    """A failure the CLI itself detected, before or after the SDK ran.

    Carries the same category vocabulary as SDK errors so one handler
    renders and codes both. ``exit_code`` overrides the category's
    default tier when a command needs to say something more specific.
    """

    def __init__(
        self,
        category: str,
        message: str,
        *,
        exit_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        if category not in CATEGORIES:
            raise ValueError(f"unknown error category {category!r}")
        self.category = category
        self.message = message
        self.exit_code = exit_code


def usage_error(message: str) -> CliError:
    """Shorthand for the most common CLI-side failure."""
    return CliError(USAGE, message)


def _sdk_error_module() -> Any:
    """Import ``dashscope.errors`` on first use, then cache the module.

    Lazy because every command module imports this one and the tables are
    only needed by the handler -- which runs once something has already
    failed, so the import is never on a ``--help`` path.
    """
    global _SDK_ERRORS  # pylint: disable=global-statement
    if _SDK_ERRORS is None:
        # pylint: disable=import-outside-toplevel
        from dashscope import errors as module

        _SDK_ERRORS = module
    return _SDK_ERRORS


def _sdk_entries() -> Dict[str, Any]:
    """The catalog's ``SDK_ERRORS`` rows, keyed by their ``name``."""
    global _SDK_ENTRY_CACHE  # pylint: disable=global-statement
    if _SDK_ENTRY_CACHE is None:
        # pylint: disable=import-outside-toplevel
        from dashscope import _error_registry as registry

        _SDK_ENTRY_CACHE = {e.name: e for e in registry.SDK_ERRORS}
    return _SDK_ENTRY_CACHE


def _public_table() -> Tuple[Tuple[type, str], ...]:
    """``(exception class, category)`` for every class the catalog names.

    Read out of ``PUBLIC_ERRORS`` rather than spelled out here, so the
    table cannot drift from a regenerated catalog. Rows sharing an
    ``error_code`` (400, 401 and 429 each have several) collapse to one
    entry: the catalog already guarantees they name the same class, and
    the same status gives them the same category.
    """
    global _PUBLIC_TABLE  # pylint: disable=global-statement
    if _PUBLIC_TABLE is None:
        # pylint: disable=import-outside-toplevel
        from dashscope import _error_registry as registry

        sdk_errors = _sdk_error_module()
        rows: list = []
        seen = set()
        for entry in registry.PUBLIC_ERRORS:
            cls = getattr(sdk_errors, entry.error_code)
            if cls in seen:
                continue
            seen.add(cls)
            # An undeclared status would mean the catalog grew a tier
            # the CLI has no opinion about yet; ``server`` is the honest
            # "not your bug" default, and the coverage test makes sure
            # this branch is never actually taken.
            rows.append((cls, _STATUS_CATEGORY.get(entry.status_code, SERVER)))
        # Base last: every row above subclasses it, so it only catches a
        # status error the catalog left unclaimed.
        rows.append((sdk_errors.APIStatusError, SERVER))
        _PUBLIC_TABLE = tuple(rows)
    return _PUBLIC_TABLE


def _unique_public_messages() -> Dict[str, str]:
    """``error_code`` -> catalog text, only where one row claims it.

    ``BadRequestError`` / ``AuthenticationError`` / ``RateLimitError``
    each back several catalog rows with different wording, so naming the
    class does not identify the entry and no text is offered -- the same
    rule ``dashscope.errors._catalog_message`` applies.
    """
    global _PUBLIC_MESSAGES  # pylint: disable=global-statement
    if _PUBLIC_MESSAGES is None:
        # pylint: disable=import-outside-toplevel
        from collections import Counter

        from dashscope import _error_registry as registry

        counts = Counter(e.error_code for e in registry.PUBLIC_ERRORS)
        _PUBLIC_MESSAGES = {
            e.error_code: e.error_msg
            for e in registry.PUBLIC_ERRORS
            if counts[e.error_code] == 1
        }
    return _PUBLIC_MESSAGES


_SDK_ERRORS: Any = None
_SDK_ENTRY_CACHE: Optional[Dict[str, Any]] = None
_PUBLIC_TABLE: Optional[Tuple[Tuple[type, str], ...]] = None
_PUBLIC_MESSAGES: Optional[Dict[str, str]] = None


class Diagnosis(NamedTuple):
    """Everything the handler needs to render and code one failure.

    A ``NamedTuple`` so the positional contract stays: ``classify(exc)[0]``
    is still the category and ``[1]`` still the message, with the
    catalog-sourced extras trailing. The catalog's ``allow_retry`` is
    deliberately not carried: the category already encodes it
    (``usage``/``auth``/``notfound`` mean "do not retry",
    ``ratelimit``/``network``/``server`` mean "maybe later"), and a
    second source of the same fact is a second thing to keep in sync.
    """

    category: str
    message: str
    solution: Optional[str] = None
    request_id: Optional[str] = None


def _message(exc: BaseException) -> str:
    """The exception's own text, undecorated.

    ``DashScopeError.__str__`` appends ``| code=... | status_code=...``,
    which reads well in a traceback and is noise on a one-line CLI error;
    ``.message`` is the service's wording and nothing else. ``str()`` is
    only the fallback for exceptions outside the SDK family.
    """
    message = getattr(exc, "message", None)
    return str(exc) if message is None else message


def _catalog_message(exc: BaseException) -> str:
    """Registry text for a gateway failure that arrived with no message.

    Only a last resort -- the service's wording always wins. Placeholders
    stay verbatim for the same reason they do in ``dashscope.errors``:
    nothing here knows the ``{resource}`` / ``{limit}`` values, and
    stripping them would hide the field the catalog declares.
    """
    return _unique_public_messages().get(type(exc).__name__, "")


def classify(exc: BaseException) -> Diagnosis:
    """Return the :class:`Diagnosis` for *exc*.

    The message is the SDK/server text verbatim; nothing is re-worded.
    """
    if isinstance(exc, CliError):
        return Diagnosis(exc.category, exc.message)

    sdk_errors = _sdk_error_module()

    # isinstance walk, not a dict lookup: ContentFilterError subclasses
    # BadRequestError, and the catalog names only the parent -- which is
    # exactly what makes the refinement legal, and why the subclass has
    # to inherit the parent's tier rather than be listed on its own.
    for klass, category in _public_table():
        if isinstance(exc, klass):
            return Diagnosis(
                category,
                _message(exc) or _catalog_message(exc),
                request_id=_request_id(exc),
            )

    if isinstance(exc, sdk_errors.SdkError):
        code = getattr(exc, "code", None) or ""
        entry = _sdk_entries().get(code)
        return Diagnosis(
            _SDK_CODE_CATEGORY.get(code, SERVER),
            _message(exc) or (entry.message if entry is not None else ""),
            solution=entry.solution if entry is not None else None,
            request_id=_request_id(exc),
        )

    if isinstance(exc, sdk_errors.DashScopeError):
        return Diagnosis(
            SERVER,
            _message(exc) or _catalog_message(exc),
            request_id=_request_id(exc),
        )

    # resolve_config rejects an invalid protocol/region/timeout with a
    # plain ValueError: that is a bad command line, not a broken service.
    if isinstance(exc, ValueError):
        return Diagnosis(USAGE, str(exc))
    return Diagnosis(SERVER, str(exc) or exc.__class__.__name__)


def exit_code_for(category: str) -> int:
    return _CATEGORY_EXIT[category]


def _request_id(exc: BaseException) -> Optional[str]:
    return getattr(exc, "request_id", None)


def print_error(
    diagnosis: Diagnosis,
    *,
    console: Console,
    verbose: bool = False,
    exc: Optional[BaseException] = None,
) -> None:
    """Render ``Error [category]: message`` on stderr.

    Markup is escaped because a server message may contain brackets,
    which Rich would otherwise read as a style tag and drop from the
    output. Wrapping and auto-highlighting are off for the same reason:
    the message must arrive byte-for-byte, not re-flowed to the terminal
    width or restyled around the numbers in it. Everything goes to stderr
    so ``--output json`` on stdout stays machine-readable.

    The catalog's ``solution`` gets its own dim line rather than being
    appended to the message: it is the registry's wording, not the
    service's, and merging them would make it impossible to tell a
    support ticket which half came from the gateway.
    """
    console.print(
        f"[red]Error[/red] [yellow]\\[{escape(diagnosis.category)}][/yellow]: "
        f"{escape(diagnosis.message)}",
        highlight=False,
        soft_wrap=True,
    )
    if diagnosis.request_id:
        console.print(
            f"[dim]request_id: {escape(str(diagnosis.request_id))}[/dim]",
            highlight=False,
            soft_wrap=True,
        )
    if diagnosis.solution:
        console.print(
            f"[dim]solution: {escape(diagnosis.solution)}[/dim]",
            highlight=False,
            soft_wrap=True,
        )
    if verbose and exc is not None:
        console.print(
            "[dim]"
            + escape(
                "".join(
                    traceback.format_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                    ),
                ),
            )
            + "[/dim]",
            highlight=False,
            soft_wrap=True,
        )


def _verbose_from(ctx: typer.Context) -> bool:
    """Best-effort verbose lookup that works before ``ctx.obj`` exists.

    The root callback runs inside the group's own ``invoke``, so an
    invalid global option raises before ``ctx.obj`` was ever assigned.
    """
    obj = getattr(ctx, "obj", None)
    if obj is not None:
        return bool(getattr(obj, "verbose", False))
    return bool(ctx.params.get("verbose"))


def _error_console() -> Console:
    return Console(
        stderr=True,
        no_color=bool(os.environ.get("NO_COLOR")),
    )


try:
    # ``typer.core.TyperGroup`` is an internal API and the dependency pin
    # is only ``typer>=0.9``, so fall back to a plain click group rather
    # than failing to import the whole CLI if a future typer moves it.
    # Losing the subclass only costs Typer's rich-help formatting; the
    # error mapping below is what actually matters and survives either way.
    from typer.core import TyperGroup as _GroupBase
except ImportError:  # pragma: no cover
    from click import Group as _GroupBase


class ErrorHandlingGroup(_GroupBase):
    """Root command group that turns every exception into a tier + line.

    A Typer callback cannot do this: click does not route command-body
    exceptions back through it. Handling only in ``main()`` is not enough
    either, because tests (and embedders) invoke the app object directly
    and would see a traceback. Overriding the group's ``invoke`` covers
    both paths at every nesting depth.
    """

    def invoke(self, ctx: typer.Context):
        try:
            return super().invoke(ctx)
        except (SystemExit, typer.Exit, KeyboardInterrupt):
            raise
        except _CLICK_PASSTHROUGH:
            # click already rendered a usage message and chose exit 2.
            raise
        except BaseException as exc:  # pylint: disable=broad-except
            diagnosis = classify(exc)
            print_error(
                diagnosis,
                console=_error_console(),
                verbose=_verbose_from(ctx),
                exc=exc,
            )
            obj = getattr(ctx, "obj", None)
            if obj is not None:
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # pylint: disable=broad-except
                        pass
            override = getattr(exc, "exit_code", None)
            sys.exit(
                override
                if override is not None
                else exit_code_for(diagnosis.category),
            )


def _click_passthrough() -> Tuple[type[BaseException], ...]:
    import click  # pylint: disable=import-outside-toplevel

    classes = [click.exceptions.Exit, click.UsageError, click.Abort]
    # typer.Exit subclasses click.exceptions.Exit on modern typer; keep
    # both spellings so an older typer still passes usage errors through.
    typer_exit = getattr(typer, "Exit", None)
    if typer_exit is not None and typer_exit not in classes:
        classes.append(typer_exit)
    return tuple(classes)


_CLICK_PASSTHROUGH: Tuple[type[BaseException], ...] = _click_passthrough()
