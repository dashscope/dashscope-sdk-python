# -*- coding: utf-8 -*-
"""OpenAI-compatible ``chat.completions.create`` instrumentation."""
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import os
import random
import threading
import time
import types
from typing import Any, Dict, Optional
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace.status import Status, StatusCode

from ....common.log import logger
from ..genai._core import (  # noqa: E501
    GENAI_AVAILABLE,
    get_handler,
)
from ..genai.messages import (  # noqa: E501  # pylint: disable=line-too-long
    openai_chat_messages_to_input_messages,
    openai_completion_to_output_messages,
    unwrap_openai_completion,
)
from ..tracing import (
    is_tracing_enabled,
    log_trace_id,
)

# Idempotency sentinel on the completions resource — owned by this module (
# cf. ``trace_tool``).
_OPENAI_COMPLETIONS_PATCHED_ATTR = "_agentic_rl_genai_openai_patched"

_DEBUG_LLM_OUTPUT = os.environ.get(
    "AGENTIC_RL_DEBUG_LLM_OUTPUT",
    "",
).lower() in ("1", "true", "yes", "y")
_DEBUG_TRACE_CLIENT_BRANCH = os.environ.get(
    "AGENTIC_RL_DEBUG_TRACE_CLIENT_BRANCH",
    "",
).lower() in (
    "1",
    "true",
    "yes",
    "y",
)


def _debug_sample_rate() -> float:
    raw = os.environ.get("AGENTIC_RL_DEBUG_TRACE_CLIENT_SAMPLE_RATE", "1.0")
    try:
        v = float(raw)
    except Exception:
        return 1.0
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


_DEBUG_TRACE_CLIENT_SAMPLE_RATE = _debug_sample_rate()


def _trace_id_hex_best_effort() -> Optional[str]:
    try:
        from opentelemetry.trace import format_trace_id

        ctx = otel_trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format_trace_id(ctx.trace_id)
    except Exception:
        return None
    return None


def _loop_id_best_effort() -> Optional[int]:
    try:
        loop = asyncio.get_running_loop()
        return id(loop)
    except Exception:
        return None


def _dbg_trace_client(event: str, **fields: Any) -> None:
    """Lightweight branch logging for diagnosing trace_client hangs.

    Enabled via env: AGENTIC_RL_DEBUG_TRACE_CLIENT_BRANCH=1
    """
    if not _DEBUG_TRACE_CLIENT_BRANCH:
        return
    if _DEBUG_TRACE_CLIENT_SAMPLE_RATE < 1.0:
        try:
            if random.random() >= _DEBUG_TRACE_CLIENT_SAMPLE_RATE:
                return
        except Exception:
            return
    try:
        base = {
            "event": event,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "loop_id": _loop_id_best_effort(),
            "trace_id": _trace_id_hex_best_effort(),
        }
        base.update(fields)
        # Stable, grep-friendly single line.
        logger.info("[trace_client_branch] %s", base)
    except Exception:
        return


def _orig_create_is_coroutine_function(orig_create: Any) -> bool:
    """True when *orig_create* unwraps to an ``async def`` (the real OpenAI
    async API).

    ``openai-python`` wraps ``AsyncCompletions.create`` with decorators;
    the bound method's ``__func__`` is often a plain ``def`` forwarding
    wrapper, so ``inspect.iscoroutinefunction(completions.create)`` is
    **False** even for ``AsyncOpenAI``. Follow ``functools.wraps`` /
    ``__wrapped__`` via :func:`inspect.unwrap` before calling
    :func:`inspect.iscoroutinefunction`.
    """
    try:
        target = inspect.unwrap(orig_create)
    except (ValueError, TypeError, AttributeError):
        target = orig_create
        if inspect.ismethod(target):
            try:
                target = inspect.unwrap(target.__func__)
            except (ValueError, TypeError, AttributeError):
                target = target.__func__
    return bool(inspect.iscoroutinefunction(target))


class _SyncCreateCompletionAdapter:
    """Let a sync ``create`` result work when upstream incorrectly
    ``await``s it.

    Some stacks (e.g. LangGraph + ``ChatOpenAI`` async paths) end up doing
    ``await completions.create(...)`` even when ``create`` is the
    **synchronous** OpenAI client method. A plain ``ChatCompletion`` is not
    awaitable. This type delegates attribute access to the real completion,
    implements ``__await__``, and supplies ``parse`` / ``parsed`` shims for
    HTTP-wrapper-shaped call sites. ``await`` resolves to **this adapter** (
    not only the inner completion) so chained ``.parse()`` after ``await``
    remains valid.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    @property
    def parsed(self) -> Any:
        """HTTP wrappers expose ``.parsed``; raw ``ChatCompletion`` does
        not."""
        inner = object.__getattribute__(self, "_inner")
        if hasattr(inner, "parsed"):
            return getattr(inner, "parsed")
        return inner

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        """HTTP wrappers expose ``parse()``; raw ``ChatCompletion`` does
        not."""
        inner = object.__getattribute__(self, "_inner")
        parse_fn = getattr(inner, "parse", None)
        if callable(parse_fn):
            return parse_fn(*args, **kwargs)
        return inner

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_inner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_inner"), name, value)

    def __repr__(self) -> str:
        inner = object.__getattribute__(self, "_inner")
        return f"_SyncCreateCompletionAdapter({inner!r})"

    def __await__(self) -> Any:
        # Return *this adapter* after await so callers still have ``.parse`` /
        # ``.parsed`` shims. Returning only ``_inner`` breaks stacks that chain
        # ``await create()`` then ``.parse()`` (expects LegacyAPIResponse
        # shape).
        outer = self

        async def _done() -> Any:
            return outer

        return _done().__await__()


def _sync_create_return_value(unwrapped: Any) -> Any:
    """Normalize sync ``create`` return for both sync use and mistaken
    ``await``."""
    if unwrapped is None:
        return None
    return _SyncCreateCompletionAdapter(unwrapped)


def _fill_llm_invocation_from_openai_kwargs(
    inv: Any,
    kwargs: Dict[str, Any],
    provider: str,
) -> None:
    inv.provider = provider
    inv.request_model = kwargs.get("model")
    messages = kwargs.get("messages")
    if messages is not None:
        inv.input_messages = openai_chat_messages_to_input_messages(messages)
    inv.temperature = kwargs.get("temperature")
    inv.top_p = kwargs.get("top_p")
    inv.max_tokens = kwargs.get("max_tokens")
    stop = kwargs.get("stop")
    if stop is not None:
        inv.stop_sequences = [stop] if isinstance(stop, str) else list(stop)
    inv.seed = kwargs.get("seed")
    freq = kwargs.get("frequency_penalty")
    if freq is not None:
        inv.frequency_penalty = freq
    pres = kwargs.get("presence_penalty")
    if pres is not None:
        inv.presence_penalty = pres


def _apply_openai_completion_to_invocation(inv: Any, completion: Any) -> None:
    # Unwrap wrapper response objects so response attributes are populated.
    unwrapped = unwrap_openai_completion(completion)
    inv.response_model_name = getattr(unwrapped, "model", None)
    inv.response_id = getattr(unwrapped, "id", None)
    usage = getattr(unwrapped, "usage", None)
    if usage is not None:
        inv.input_tokens = getattr(usage, "prompt_tokens", None)
        inv.output_tokens = getattr(usage, "completion_tokens", None)
    inv.output_messages = openai_completion_to_output_messages(unwrapped)
    if _DEBUG_LLM_OUTPUT:
        try:
            out = getattr(inv, "output_messages", None)
            logger.info(
                "[debug_llm_output] completion_type=%s unwrapped_type=%s "
                "output_messages_len=%s",
                type(completion).__name__,
                type(unwrapped).__name__,
                (len(out) if isinstance(out, list) else "n/a"),
            )
        except Exception:
            pass


def _best_effort_span(inv: Any) -> Optional[Any]:
    """Get a span-like object to annotate, best-effort."""
    try:
        span = getattr(inv, "span", None)
        if span is not None:
            return span
    except Exception:
        pass
    try:
        return otel_trace.get_current_span()
    except Exception:
        return None


def _best_effort_mark_error(inv: Any, exc: BaseException) -> None:
    """Make failures visible without breaking business logic."""
    span = _best_effort_span(inv)
    if span is None:
        return
    try:
        if (
            callable(getattr(span, "is_recording", None))
            and not span.is_recording()
        ):
            return
    except Exception:
        pass
    # Status may be ignored by some backends; exception events are more
    # reliable.
    try:
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        pass
    try:
        span.record_exception(exc)
    except Exception:
        pass
    try:
        span.set_attribute("error.type", type(exc).__qualname__)
        span.set_attribute("error.message", str(exc))
    except Exception:
        pass


def _best_effort_cm_exit(cm: Any, exc: Optional[BaseException]) -> None:
    """Best-effort close a context manager with explicit exc info.

    We avoid relying on sys.exc_info() at arbitrary frames; callers supply the
    actual exception instance (or None) that should be reflected in __exit__.
    """
    try:
        if exc is None:
            cm.__exit__(None, None, None)
        else:
            cm.__exit__(type(exc), exc, exc.__traceback__)
    except Exception:
        return


async def _await_and_apply(inv: Any, awaitable: Any) -> Any:
    """Await an awaitable completion and then populate invocation.

    Some OpenAI-compatible clients expose a sync ``create`` method but
    return an awaitable (coroutine/future). We must not treat that awaitable
    as the final completion object; instead we attach instrumentation after
    awaiting.
    """
    try:
        completion = await awaitable
        _apply_openai_completion_to_invocation(inv, completion)
        return completion
    except Exception as e:
        _best_effort_mark_error(inv, e)
        raise


def _run_coroutine_blocking(coro: Any) -> Any:
    """Run *coro* to completion without calling ``asyncio.run`` under a
    running loop.

    Sync ``create`` sometimes returns an awaitable while the caller stack
    may already hold an asyncio loop (e.g. LangChain inside async).
    ``asyncio.run``  would then raise; we isolate ``asyncio.run`` in a
    one-off thread instead.
    OTel context is attached in the worker so any span logic there stays
    under the same trace as the caller (``contextvars`` do not propagate to
    thread pools).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _dbg_trace_client("run_coroutine_blocking:no_running_loop")
        return asyncio.run(coro)

    otel_ctx = otel_context.get_current()

    def _runner() -> Any:
        _dbg_trace_client("run_coroutine_blocking:thread_runner_enter")
        token = otel_context.attach(otel_ctx)
        try:
            return asyncio.run(coro)
        finally:
            otel_context.detach(token)
            _dbg_trace_client("run_coroutine_blocking:thread_runner_exit")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        _dbg_trace_client("run_coroutine_blocking:submit_thread")
        return ex.submit(_runner).result()


def _resolve_completions(client: Any) -> Any:
    """Return the ``chat.completions`` resource from *client*.

    Accepts two shapes:

    * A full OpenAI client (``openai.OpenAI`` / ``openai.AsyncOpenAI``) that
      exposes ``.chat.completions``.
    * A completions resource directly (e.g.
    ``langchain_openai.ChatOpenAI.client`` which is already ``openai.OpenAI(
    ...).chat.completions``). Detected by the presence of a callable
    ``.create`` and the *absence* of a ``.chat`` attribute — the contract
    LangChain has kept since 0.1.x.

    Returns ``None`` when neither shape is matched.
    """
    # Shape 2: already a completions object
    if callable(getattr(client, "create", None)) and not hasattr(
        client,
        "chat",
    ):
        return client
    # Shape 1: full client with .chat.completions
    chat = getattr(client, "chat", None)
    completions = (
        getattr(chat, "completions", None) if chat is not None else None
    )
    return completions


def instrument_openai_chat_completions(  # pylint: disable=too-many-statements
    client: Any,
    *,
    provider: str = "openai",
    handler: Any = None,
) -> Any:
    """
    Monkey-patch the ``create`` method on an OpenAI-compatible completions
    resource to emit a GenAI ``llm`` span on every call.

    *client* may be:

    * A full OpenAI client instance (``openai.OpenAI`` /
    ``openai.AsyncOpenAI``).
    * A completions resource directly (e.g. ``ChatOpenAI.client`` from
      ``langchain-openai``, which is ``openai.OpenAI(...).chat.completions``).

    The patch is idempotent — calling this function twice on the same object is
    safe (the second call is a no-op).

    No-op when tracing is disabled or ``loongsuite-util-genai`` is not
    installed. Returns *client* unchanged so the call can be chained.
    """
    if not is_tracing_enabled():
        return client
    if not GENAI_AVAILABLE:
        logger.warning(
            "GenAI tracing requested but loongsuite-util-genai is not "
            "installed. pip install loongsuite-util-genai",
        )
        return client

    completions = _resolve_completions(client)
    if completions is None:
        logger.warning(
            "instrument_openai_chat_completions: could not locate a "
            "completions resource on %r — expected .chat.completions or a "
            "direct completions object",
            type(client),
        )
        return client

    if getattr(completions, _OPENAI_COMPLETIONS_PATCHED_ATTR, False):
        return client

    orig_create = completions.create
    h = handler if handler is not None else get_handler()

    if _orig_create_is_coroutine_function(orig_create):

        async def async_wrapper(  # pylint: disable=too-many-statements
            _self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with h.llm() as inv:
                log_trace_id("llm")
                _fill_llm_invocation_from_openai_kwargs(inv, kwargs, provider)
                try:
                    t0 = time.perf_counter()
                    _dbg_trace_client(
                        "llm_async_wrapper:enter",
                        provider=provider,
                        create_is_coro=True,
                    )
                    completion = await orig_create(*args, **kwargs)
                    _dbg_trace_client(
                        "llm_async_wrapper:orig_create_done",
                        elapsed_ms=int((time.perf_counter() - t0) * 1000),
                        completion_type=type(completion).__name__,
                    )
                    _apply_openai_completion_to_invocation(inv, completion)
                    # Return the client's native response object (often
                    # ``LegacyAPIResponse`` or similar) so LangChain /
                    # OpenAI stacks can call ``.parse()`` / access
                    # HTTP-wrapper fields. GenAI attributes are filled from
                    # the same object via ``unwrap_openai_completion`` inside
                    # ``_apply_openai_completion_to_invocation``.
                    _dbg_trace_client(
                        "llm_async_wrapper:return",
                        completion_type=type(completion).__name__,
                        return_has_parse=callable(
                            getattr(completion, "parse", None),
                        ),
                    )
                    return completion
                except Exception as e:
                    _best_effort_mark_error(inv, e)
                    _dbg_trace_client(
                        "llm_async_wrapper:error",
                        err_type=type(e).__name__,
                        err=str(e),
                    )
                    raise

        completions.create = types.MethodType(async_wrapper, completions)
    else:

        def sync_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            _dbg_trace_client(
                "llm_sync_wrapper:enter",
                provider=provider,
                create_is_coro=False,
            )
            # Some OpenAI-compatible stacks expose a *sync* create() that
            # returns an awaitable. In that case we must keep the invocation
            # open until the awaitable completes — but we must not block the
            # current thread or create a new event loop thread (that path
            # has been shown to hang).
            #
            # Implementation:
            # - If create() returns a normal completion: do the usual sync
            # path.
            # - If create() returns an awaitable: manually enter/exit the
            # handler's
            #   context manager inside the returned awaitable, and attach
            #   the caller OTel context when awaiting, so spans remain
            #   parented correctly.
            cm = h.llm()
            inv = cm.__enter__()
            returned_awaitable = False
            outer_exc: Optional[BaseException] = None
            try:
                log_trace_id("llm")
                _fill_llm_invocation_from_openai_kwargs(inv, kwargs, provider)
                completion = orig_create(*args, **kwargs)
                is_awaitable = bool(inspect.isawaitable(completion))
                _dbg_trace_client(
                    "llm_sync_wrapper:orig_create_returned",
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                    returned_type=type(completion).__name__,
                    returned_is_awaitable=is_awaitable,
                )
                if not is_awaitable:
                    _apply_openai_completion_to_invocation(inv, completion)
                    # Wrap the original response (do not unwrap here) so
                    # ``.parse()`` and wrapper attributes remain available
                    # to LangChain / OpenAI clients.
                    _dbg_trace_client(
                        "llm_sync_wrapper:return_sync",
                        completion_type=type(completion).__name__,
                        return_has_parse=callable(
                            getattr(completion, "parse", None),
                        ),
                    )
                    return _sync_create_return_value(completion)

                otel_ctx = otel_context.get_current()
                awaitable = completion

                async def _wrapped() -> Any:
                    token = None
                    exc: Optional[BaseException] = None
                    try:
                        _dbg_trace_client(
                            "llm_sync_wrapper:awaitable_wrap:enter",
                        )
                        token = otel_context.attach(otel_ctx)
                        resolved = await _await_and_apply(inv, awaitable)
                        _dbg_trace_client(
                            "llm_sync_wrapper:awaitable_wrap:done",
                            total_elapsed_ms=int(
                                (time.perf_counter() - t0) * 1000,
                            ),
                            completion_type=type(resolved).__name__,
                            return_has_parse=callable(
                                getattr(resolved, "parse", None),
                            ),
                        )
                        return _sync_create_return_value(resolved)
                    except asyncio.CancelledError as e:
                        exc = e
                        _best_effort_mark_error(inv, e)
                        _dbg_trace_client(
                            "llm_sync_wrapper:awaitable_wrap:cancelled",
                            total_elapsed_ms=int(
                                (time.perf_counter() - t0) * 1000,
                            ),
                        )
                        raise
                    except Exception as e:
                        exc = e
                        _best_effort_mark_error(inv, e)
                        _dbg_trace_client(
                            "llm_sync_wrapper:awaitable_wrap:error",
                            err_type=type(e).__name__,
                            err=str(e),
                            total_elapsed_ms=int(
                                (time.perf_counter() - t0) * 1000,
                            ),
                        )
                        raise
                    finally:
                        # Ensure stop/end happens under the caller context.
                        _best_effort_cm_exit(cm, exc)
                        if token is not None:
                            try:
                                otel_context.detach(token)
                            except Exception:
                                pass

                returned_awaitable = True
                return _wrapped()
            except Exception as e:
                outer_exc = e
                _best_effort_mark_error(inv, e)
                _dbg_trace_client(
                    "llm_sync_wrapper:error",
                    err_type=type(e).__name__,
                    err=str(e),
                )
                raise
            finally:
                # If we returned an awaitable above, its finally will __exit__.
                if not returned_awaitable:
                    _best_effort_cm_exit(cm, outer_exc)

        completions.create = types.MethodType(sync_wrapper, completions)

    object.__setattr__(completions, _OPENAI_COMPLETIONS_PATCHED_ATTR, True)
    return client
