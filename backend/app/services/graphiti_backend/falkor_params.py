"""Keep FalkorDB writes within the property types FalkorDB actually accepts.

FalkorDB rejects a whole query when any property value is a map, or an array
containing maps: *"Property values can only be of primitive types or arrays of
primitive types"*.

graphiti stores the **raw** LLM attribute dict (``_extract_entity_attributes``
returns the model's reply, not a validated dump), so any nested object an LLM
invents — including keys outside the ontology — reaches the driver verbatim and
fails the entire episode. Rather than trying to anticipate every shape a model
might emit, this module enforces FalkorDB's invariant at the write boundary:
non-primitive property values become JSON text.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.falkor")

_PRIMITIVES = (str, int, float, bool, bytes, datetime, date)


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, _PRIMITIVES)


def _is_acceptable_property(value: Any) -> bool:
    """Return whether FalkorDB will accept ``value`` as a property value."""

    if _is_primitive(value):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_primitive(item) for item in value)
    return False


def sanitize_property_value(value: Any, *, context: str = "") -> Any:
    if _is_acceptable_property(value):
        return value
    logger.warning(
        "Serializing non-primitive FalkorDB property%s to JSON text (%s)",
        f" {context}" if context else "",
        type(value).__name__,
    )
    return json.dumps(value, ensure_ascii=False, default=str)


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sanitize_property_value(value, context=key) for key, value in row.items()
    }


def sanitize_query_params(params: dict[str, Any]) -> dict[str, Any]:
    """Sanitize the property-bearing values in one query's parameters.

    Parameters are either a row map (``SET n = $props``) or a list of row maps
    fed through ``UNWIND``. Only values *inside* those rows become properties,
    so the outer structure is left alone.
    """

    sanitized: dict[str, Any] = {}
    for name, value in params.items():
        if isinstance(value, dict):
            sanitized[name] = sanitize_row(value)
        elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
            sanitized[name] = [
                sanitize_row(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            sanitized[name] = value
    return sanitized


def install_param_sanitizer(driver: Any) -> Any:
    """Wrap ``driver`` so every write sanitizes its parameters first.

    Applied per instance rather than by subclassing: ``FalkorDriver.clone()``
    hard-codes its own class, so a subclass would be lost the moment graphiti
    clones the driver for another group.
    """

    if getattr(driver, "_mirofish_sanitized", False):
        return driver

    original_session = driver.session
    original_execute_query = driver.execute_query

    def session(database: str | None = None) -> Any:
        return _SanitizingSession(original_session(database))

    async def execute_query(cypher_query_: Any, **kwargs: Any) -> Any:
        routing = {"routing_": kwargs.pop("routing_")} if "routing_" in kwargs else {}
        return await original_execute_query(
            cypher_query_, **sanitize_query_params(kwargs), **routing
        )

    driver.session = session
    driver.execute_query = execute_query
    driver._mirofish_sanitized = True
    return driver


class _SanitizingSession:
    """Delegate to a driver session, sanitizing parameters on the way in."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def run(self, query: Any, **kwargs: Any) -> Any:
        if isinstance(query, list):
            query = [
                (cypher, sanitize_query_params(params or {})) for cypher, params in query
            ]
            return await self._inner.run(query)
        return await self._inner.run(query, **sanitize_query_params(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def __aenter__(self) -> "_SanitizingSession":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._inner.__aexit__(*exc_info)
