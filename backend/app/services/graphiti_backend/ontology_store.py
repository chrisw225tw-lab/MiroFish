"""Translate MiroFish's Zep ontology into graphiti types, and persist it.

Zep enforces an ontology per graph on the server. graphiti applies types
per ``add_episode`` call, so the adapter has to remember what was registered.
The registry is written to disk because ingestion can span a backend restart.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, create_model

from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.ontology")

# graphiti reserves these on its own node/edge models.
RESERVED_FIELD_NAMES = frozenset(
    {
        "uuid",
        "name",
        "group_id",
        "labels",
        "created_at",
        "summary",
        "attributes",
        "fact",
        "episodes",
        "source_node_uuid",
        "target_node_uuid",
    }
)


@dataclass(frozen=True)
class TypeSpec:
    """A serializable description of one ontology entity or edge type."""

    name: str
    description: str
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypeSpec":
        attributes = payload.get("attributes") or {}
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description") or payload["name"]),
            attributes={str(k): str(v) for k, v in attributes.items()},
        )

    def to_model(self) -> type[BaseModel]:
        """Build the pydantic model graphiti uses for attribute extraction."""

        fields: dict[str, Any] = {
            attr_name: (Optional[str], Field(default=None, description=attr_desc))
            for attr_name, attr_desc in self.attributes.items()
            if attr_name not in RESERVED_FIELD_NAMES
        }
        model = create_model(_safe_model_name(self.name), **fields)
        model.__doc__ = self.description
        return model


@dataclass(frozen=True)
class OntologySpec:
    """The full per-graph ontology, in a form that survives a restart."""

    entities: list[TypeSpec] = field(default_factory=list)
    edges: list[TypeSpec] = field(default_factory=list)
    # (source_entity_label, target_entity_label) -> edge type names
    edge_type_map: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [spec.to_dict() for spec in self.entities],
            "edges": [spec.to_dict() for spec in self.edges],
            "edge_type_map": [
                {"source": source, "target": target, "edge_types": list(edge_types)}
                for (source, target), edge_types in self.edge_type_map.items()
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OntologySpec":
        edge_type_map: dict[tuple[str, str], list[str]] = {}
        for entry in payload.get("edge_type_map") or []:
            key = (str(entry.get("source", "Entity")), str(entry.get("target", "Entity")))
            edge_type_map[key] = [str(name) for name in entry.get("edge_types") or []]
        return cls(
            entities=[TypeSpec.from_dict(item) for item in payload.get("entities") or []],
            edges=[TypeSpec.from_dict(item) for item in payload.get("edges") or []],
            edge_type_map=edge_type_map,
        )

    def as_graphiti_kwargs(self) -> dict[str, Any]:
        """Return the ``add_episode`` keyword arguments for this ontology."""

        return {
            "entity_types": {spec.name: spec.to_model() for spec in self.entities},
            "edge_types": {spec.name: spec.to_model() for spec in self.edges},
            "edge_type_map": dict(self.edge_type_map),
        }


def _safe_model_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in name).strip("_")
    return cleaned or "OntologyType"


def _field_descriptions(model: type[BaseModel]) -> dict[str, str]:
    return {
        field_name: (field_info.description or field_name)
        for field_name, field_info in model.model_fields.items()
        if field_name not in RESERVED_FIELD_NAMES
    }


def build_ontology_spec(
    entities: Mapping[str, type[BaseModel]] | None,
    edges: Mapping[str, tuple[type[BaseModel], Sequence[Any]]] | None,
) -> OntologySpec:
    """Convert the Zep ``set_ontology`` payload into an ``OntologySpec``."""

    entity_specs = [
        TypeSpec(
            name=name,
            description=(model.__doc__ or name).strip(),
            attributes=_field_descriptions(model),
        )
        for name, model in (entities or {}).items()
    ]

    edge_specs: list[TypeSpec] = []
    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for name, definition in (edges or {}).items():
        model, source_targets = definition
        edge_specs.append(
            TypeSpec(
                name=name,
                description=(model.__doc__ or name).strip(),
                attributes=_field_descriptions(model),
            )
        )
        for source_target in source_targets or []:
            key = (
                getattr(source_target, "source", None) or "Entity",
                getattr(source_target, "target", None) or "Entity",
            )
            edge_type_map.setdefault(key, [])
            if name not in edge_type_map[key]:
                edge_type_map[key].append(name)

    return OntologySpec(entities=entity_specs, edges=edge_specs, edge_type_map=edge_type_map)


class OntologyStore:
    """In-process ontology registry backed by one JSON file per graph."""

    def __init__(self, state_dir: str) -> None:
        self._state_dir = state_dir
        self._lock = threading.Lock()
        self._cache: dict[str, OntologySpec] = {}

    def _path(self, graph_id: str) -> str:
        return os.path.join(self._state_dir, f"{graph_id}_ontology.json")

    def set(self, graph_id: str, spec: OntologySpec) -> None:
        with self._lock:
            self._cache[graph_id] = spec
            os.makedirs(self._state_dir, exist_ok=True)
            path = self._path(graph_id)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(spec.to_dict(), handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)

    def get(self, graph_id: str) -> OntologySpec | None:
        with self._lock:
            cached = self._cache.get(graph_id)
            if cached is not None:
                return cached
            path = self._path(graph_id)
            if not os.path.exists(path):
                return None
            try:
                with open(path, encoding="utf-8") as handle:
                    spec = OntologySpec.from_dict(json.load(handle))
            except (OSError, ValueError, KeyError) as error:
                logger.warning(
                    "Ignoring unreadable ontology state for graph %s: %s", graph_id, error
                )
                return None
            self._cache[graph_id] = spec
            return spec

    def delete(self, graph_id: str) -> None:
        with self._lock:
            self._cache.pop(graph_id, None)
            try:
                os.remove(self._path(graph_id))
            except FileNotFoundError:
                pass
            except OSError as error:
                logger.warning(
                    "Could not remove ontology state for graph %s: %s", graph_id, error
                )

    def episode_kwargs(self, graph_id: str) -> dict[str, Any]:
        """Return ``add_episode`` type arguments, or empty when unset."""

        spec = self.get(graph_id)
        return spec.as_graphiti_kwargs() if spec else {}
