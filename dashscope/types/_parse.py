# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fault-tolerant construction of server response models.

A gateway that adds a field, renames one, or starts sending a string
where an ``int`` used to be is not an outage -- but a bare
``Model(**body)`` turns it into a :class:`pydantic.ValidationError`
escaping the SDK, which no ``except DashScopeError`` can catch. Every
server-facing model here is already ``extra="allow"``, so unknown keys
survive; what breaks it is a *typed* field whose value drifted, or a
required field the server stopped sending.

:func:`parse_model` degrades instead of raising: the offending fields
alone are repaired, the rest of the body keeps its full validation, and
the drift is recorded at ``debug`` level. ``strict=True`` opts out and
raises :class:`~dashscope.errors.APIResponseValidationError`, which is
what contract testing and CI want -- silence is the wrong default there.

Repairing only the failing fields and then re-validating the whole body
is deliberate, and is why this is not a ``model_construct`` wrapper.
``model_construct`` skips ``mode="after"`` validators, so a degraded
error response would report ``is_error=False`` and masquerade as a
success; it also leaves nested models as plain dicts, so
``completion.usage.input_tokens`` would raise ``AttributeError`` on an
object whose healthy twin has a ``Usage`` there. A second
``model_validate`` pass has neither problem, because pydantic lets
already-constructed instances through untouched. ``model_construct`` is
kept only as the last resort for a body whose individually repaired
fields still fail together, and that path re-runs the after-validators
by hand.

Values that cannot be repaired fall back to the field's default, or to
a zero value shaped like its annotation. List elements are replaced in
place rather than dropped: downstream code indexes by position (tool
call merging, embedding dimensions), so a shorter list would silently
misalign it.
"""

from __future__ import annotations

import threading
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import httpx
from pydantic import AliasChoices, BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from dashscope._logging import get_logger
from dashscope.errors import APIResponseValidationError

__all__ = ["parse_model"]

logger = get_logger("parse")

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Both caches are bounded by the model catalog, not by traffic, and are
# only ever touched on the degraded path -- a well-formed response never
# pays for an adapter.
_LOCK = threading.Lock()
_ALIAS_MAPS: Dict[type, Dict[str, str]] = {}
_ADAPTERS: Dict[Tuple[type, str], TypeAdapter] = {}

_SEQUENCE_ORIGINS = (list, tuple, set, frozenset)

# Ordered: ``bool`` must be tested before ``int``, whose subclass it is.
_ZERO_BY_TYPE: Tuple[Tuple[type, Any], ...] = (
    (bool, False),
    (int, 0),
    (float, 0.0),
    (str, ""),
)


def parse_model(
    cls: Type[_ModelT],
    data: Mapping[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> _ModelT:
    """Build ``cls`` from a server body, degrading rather than raising.

    ``strict`` raises :class:`~dashscope.errors.APIResponseValidationError`
    carrying ``response`` and the raw ``body`` instead of degrading. It
    fires even when a resource is in non-raising (``raise_on_error=False``)
    mode: a schema assertion the caller opted into is not a service
    error to be reported through the response envelope.
    """
    if not isinstance(data, Mapping):
        # There are no fields to repair, and salvage would die on
        # ``dict(data)`` with a bare TypeError or ValueError -- exactly
        # the escape this module exists to prevent.
        if strict:
            raise APIResponseValidationError(response=response, body=data)
        logger.debug(
            "response drift: %s got %s where a mapping was expected; "
            "degraded parse applied",
            cls.__name__,
            type(data).__name__,
        )
        return _zero_model(cls)
    try:
        return cls.model_validate(data)
    except ValidationError as exc:
        error_count = len(exc.errors())
        if strict:
            logger.debug(
                "response drift: %s failed strict validation (%d error(s))",
                cls.__name__,
                error_count,
            )
            raise APIResponseValidationError(
                response=response,
                body=dict(data),
                cause=exc,
            ) from exc
        salvaged = _salvage(cls, data, exc)
    logger.debug(
        "response drift: %s failed validation (%d error(s)); "
        "degraded parse applied",
        cls.__name__,
        error_count,
    )
    try:
        return cls.model_validate(salvaged)
    except ValidationError:
        # Individually valid fields that still fail together mean a
        # cross-field invariant; only constructing can keep the body.
        return _construct_degraded(cls, salvaged)


def _alias_map(cls: type) -> Dict[str, str]:
    """Map every input key a field answers to back to its field name.

    ``ValidationError`` reports the key it actually read, which for an
    aliased field is the alias (``open_source``, not the field name), so
    resolving a drift location needs the reverse of pydantic's aliasing.
    """
    with _LOCK:
        cached = _ALIAS_MAPS.get(cls)
        if cached is not None:
            return cached
        out: Dict[str, str] = {}
        for name, info in cls.model_fields.items():
            out[name] = name
            for key in _alias_keys(info):
                out.setdefault(key, name)
        _ALIAS_MAPS[cls] = out
        return out


def _alias_keys(info: FieldInfo) -> List[str]:
    alias = info.validation_alias
    if isinstance(alias, str):
        return [alias]
    if isinstance(alias, AliasChoices):
        # AliasPath entries address a nested location, not a top-level
        # key, so there is no root to map and they are skipped.
        return [c for c in alias.choices if isinstance(c, str)]
    return []


def _adapter(cls: type, name: str) -> TypeAdapter:
    key = (cls, name)
    with _LOCK:
        cached = _ADAPTERS.get(key)
        if cached is not None:
            return cached
        adapter = TypeAdapter(cls.model_fields[name].rebuild_annotation())
        _ADAPTERS[key] = adapter
        return adapter


def _salvage(
    cls: type,
    data: Mapping[str, Any],
    exc: ValidationError,
) -> Dict[str, Any]:
    """A copy of ``data`` with only the failing fields repaired.

    Error-driven rather than a sweep of every field: healthy fields and
    ``extra="allow"`` extras are carried over untouched, so a large
    well-formed payload is not re-validated piece by piece.
    """
    out = dict(data)
    fields = cls.model_fields
    aliases = _alias_map(cls)
    for err in exc.errors():
        loc = err.get("loc") or ()
        if not loc or not isinstance(loc[0], str):
            continue
        root = loc[0]
        name = aliases.get(root)
        info = fields.get(name) if name is not None else None
        if info is None:
            # Not a field this model knows about; dropping it is the
            # only repair available, and extra keys are not load-bearing.
            logger.debug(
                "response drift: %s.%s: unrecoverable location; dropped",
                cls.__name__,
                root,
            )
            out.pop(root, None)
            continue
        if err.get("type") == "missing":
            if len(loc) == 1:
                fallback = _field_fallback(info)
                # ``root`` is the key pydantic itself looked for, which
                # for an aliased field is the alias -- writing the field
                # name would not populate it.
                out[root] = fallback
                _log_drift(cls, name, "missing", None, fallback)
            elif root in out:
                # The gap is nested: zeroing the whole root would throw
                # away the siblings pydantic already accepted, so repair
                # at the depth the error actually names.
                original = out[root]
                repaired = _salvage_value(cls, name, info, original, loc[1:])
                out[root] = repaired
                _log_drift(cls, name, "missing", original, repaired, loc[1:])
            continue
        if root not in out:
            continue
        original = out[root]
        repaired = _salvage_value(cls, name, info, original, loc[1:])
        out[root] = repaired
        _log_drift(cls, name, err.get("type"), original, repaired, loc[1:])
    return out


def _salvage_value(
    cls: type,
    name: str,
    info: FieldInfo,
    raw: Any,
    sub_loc: Tuple[Any, ...],
) -> Any:
    """Repair one field's value, keeping its shape where possible.

    ``sub_loc`` is the remainder of a single error path below the root
    key -- one error at a time, so two errors on the same list arrive as
    two calls, each repairing one index of the value the first returned.
    """
    if isinstance(raw, BaseModel):
        # A sibling error under the same root already repaired this
        # field into a model instance, and pydantic passes instances
        # through untouched. Falling through to the fallback below would
        # discard that repair -- and with it every healthy subfield.
        return raw
    annotation = _unwrap_optional(info.rebuild_annotation())
    origin = get_origin(annotation)

    if origin in _SEQUENCE_ORIGINS:
        args = get_args(annotation)
        index = sub_loc[0] if sub_loc else None
        if isinstance(raw, (list, tuple)) and args and isinstance(index, int):
            return _salvage_sequence(cls, name, args[0], raw, index)
        # The complaint is about the list itself, not one element.
        return _field_fallback(info)

    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and isinstance(raw, Mapping)
    ):
        return _salvage_model(annotation, raw)

    return _field_fallback(info)


def _salvage_sequence(
    cls: type,
    name: str,
    item_type: Any,
    raw: Any,
    index: int,
) -> Any:
    """Repair one element of a list, keeping every index in place.

    Only the position pydantic named is touched, so a 1000-dimension
    embedding costs one repair rather than a thousand validations, and
    a caller that indexes by position stays aligned.
    """
    out = list(raw)
    if 0 <= index < len(out):
        out[index] = _salvage_item(cls, name, item_type, out[index])
    return out


def _salvage_item(cls: type, name: str, item_type: Any, raw: Any) -> Any:
    if isinstance(raw, BaseModel):
        # Already repaired by an earlier error on this same list; the
        # zero below would replace a good element with an empty one.
        return raw
    item_type = _unwrap_optional(item_type)
    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
        if isinstance(raw, Mapping):
            return _salvage_model(item_type, raw)
        return _zero(item_type)
    try:
        return TypeAdapter(item_type).validate_python(raw)
    except ValidationError as exc:
        errors = exc.errors()
        logger.debug(
            "response drift: %s.%s[]: expected %s, got %s (%s)",
            cls.__name__,
            name,
            _type_name(item_type),
            type(raw).__name__,
            errors[0].get("type") if errors else "invalid",
        )
        return _zero(item_type)


def _salvage_model(model_cls: Type[BaseModel], raw: Mapping[str, Any]) -> Any:
    """Validate a nested body, degrading it the same way the parent did."""
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        salvaged = _salvage(model_cls, raw, exc)
    try:
        return model_cls.model_validate(salvaged)
    except ValidationError:
        return _construct_degraded(model_cls, salvaged)


def _field_fallback(info: FieldInfo) -> Any:
    if not info.is_required():
        return info.get_default(call_default_factory=True)
    return _zero(info.rebuild_annotation())


def _zero(annotation: Any) -> Any:
    """An empty value shaped like ``annotation``.

    A required nested model becomes a real instance built from zeroed
    required fields, so ``choice.message.role`` still resolves on a
    degraded object instead of raising ``AttributeError``.
    """
    annotation = _unwrap_optional(annotation)
    if annotation is None or annotation is Any:
        return None
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _zero_model(annotation)
        for candidate, empty in _ZERO_BY_TYPE:
            if issubclass(annotation, candidate):
                return empty
        if issubclass(annotation, (list, tuple, set, frozenset, dict)):
            return annotation()
    return _zero_parameterised(get_origin(annotation))


def _zero_parameterised(origin: Any) -> Any:
    """Zero for ``List[X]`` / ``Dict[K, V]``, where origin is not a type."""
    if origin in _SEQUENCE_ORIGINS:
        return []
    if origin is dict:
        return {}
    return None


def _zero_model(model_cls: Type[BaseModel]) -> Any:
    required = {
        name: _zero(info.rebuild_annotation())
        for name, info in model_cls.model_fields.items()
        if info.is_required()
    }
    try:
        return model_cls.model_validate(required)
    except ValidationError:
        return model_cls.model_construct(**required)


def _construct_degraded(
    cls: Type[_ModelT],
    salvaged: Mapping[str, Any],
) -> _ModelT:
    """Last resort: build without validation, then restore invariants.

    ``model_construct`` ignores aliases, so alias-keyed entries are
    renamed to field names first; anything left over is an extra and
    rides along, which ``extra="allow"`` keeps readable. A key that
    became a field is not *also* an extra -- carrying both makes
    ``model_dump()`` emit the value twice (``open_source`` alongside the
    wire spelling ``opensource``), which then fails to round-trip into an
    ``extra="forbid"`` model.
    """
    fields = cls.model_fields
    aliases = _alias_map(cls)
    kwargs: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    consumed: Set[str] = set()
    for key, value in salvaged.items():
        if key in fields:
            kwargs[key] = value
            consumed.add(key)
    for key, value in salvaged.items():
        name = aliases.get(key)
        if name is not None and name not in kwargs:
            kwargs[name] = value
            consumed.add(key)
    for key, value in salvaged.items():
        if key not in consumed:
            extras[key] = value
    instance = cls.model_construct(**kwargs, **extras)
    _rerun_after_validators(cls, instance)
    return instance


def _rerun_after_validators(cls: type, instance: BaseModel) -> None:
    """Re-apply ``mode="after"`` validators ``model_construct`` skipped.

    Without this a degraded body carrying a business ``code`` would
    report ``is_error=False`` and read as a success. ``before`` and
    ``wrap`` validators are not re-run: they transform raw input, and
    feeding them an already-built instance is not what they were
    written for.
    """
    decorators = getattr(cls, "__pydantic_decorators__", None)
    validators = getattr(decorators, "model_validators", None)
    if not validators:
        return
    for name, decorator in validators.items():
        mode = getattr(getattr(decorator, "info", None), "mode", None)
        if mode != "after":
            logger.debug(
                "response drift: %s: skipped %s validator %s on "
                "degraded instance",
                cls.__name__,
                mode,
                name,
            )
            continue
        func = getattr(decorator, "func", None)
        if func is None:
            continue
        try:
            func(instance)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "response drift: %s: after-validator %s failed on "
                "degraded instance: %s",
                cls.__name__,
                name,
                type(exc).__name__,
            )


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _log_drift(
    cls: type,
    name: str,
    reason: Optional[str],
    raw: Any,
    repaired: Any,
    sub_locs: Tuple[Any, ...] = (),
) -> None:
    """Record one repaired field.

    The raw *value* is never logged -- a response body carries user
    prompts and model output, and the redaction filter is key-oriented
    so it would not catch them. The value's type says enough to
    recognise the drift.
    """
    path = ".".join(str(part) for part in (name,) + tuple(sub_locs))
    if sub_locs:
        # The leaf was already logged, with its own type, by the nested
        # salvage; ``raw`` here is the enclosing container, so reporting
        # its type against the leaf's reason would misdescribe the drift.
        logger.debug(
            "response drift: %s.%s: repaired at %s (%s)",
            cls.__name__,
            name,
            path,
            reason or "invalid",
        )
        return
    logger.debug(
        "response drift: %s.%s: %s (got %s); replaced with %s",
        cls.__name__,
        path,
        reason or "invalid",
        type(raw).__name__ if raw is not None else "absent",
        type(repaired).__name__,
    )
