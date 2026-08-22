"""Zep-shaped read models built from graphiti-core entities.

MiroFish's services read Zep SDK objects by attribute (``uuid_``, ``fact``,
``processed`` ...). Mapping graphiti results into the same attribute surface
keeps every caller — and ``utils/zep_paging`` — unchanged across backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode


@dataclass(frozen=True)
class GraphSummary:
    """Return shape for ``graph.create`` / ``graph.get``."""

    graph_id: str
    name: str | None = None
    description: str | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class NodeView:
    """Zep ``EntityNode`` surface."""

    uuid_: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    created_at: str | None = None

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass(frozen=True)
class EdgeView:
    """Zep ``EntityEdge`` surface."""

    uuid_: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: dict[str, Any]
    episodes: list[str] = field(default_factory=list)
    created_at: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None

    @property
    def uuid(self) -> str:
        return self.uuid_

    @property
    def fact_type(self) -> str:
        return self.name


@dataclass(frozen=True)
class EpisodeView:
    """Zep ``Episode`` surface.

    ``processed`` is always True: graphiti extracts synchronously inside
    ``add_episode``, so an episode that exists has already been ingested.
    """

    uuid_: str
    content: str
    created_at: str | None = None
    source_description: str = ""
    processed: bool = True

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass(frozen=True)
class SearchResults:
    """Zep ``graph.search`` surface."""

    edges: list[EdgeView] = field(default_factory=list)
    nodes: list[NodeView] = field(default_factory=list)
    episodes: list[EpisodeView] = field(default_factory=list)


@dataclass(frozen=True)
class RawResponse:
    """``with_raw_response`` surface: payload plus cursor headers."""

    data: list[Any]
    headers: dict[str, str]


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def to_node_view(node: EntityNode) -> NodeView:
    return NodeView(
        uuid_=node.uuid,
        name=node.name or "",
        labels=list(node.labels or []),
        summary=node.summary or "",
        attributes=dict(node.attributes or {}),
        created_at=_iso(node.created_at),
    )


def to_edge_view(edge: EntityEdge) -> EdgeView:
    return EdgeView(
        uuid_=edge.uuid,
        name=edge.name or "",
        fact=edge.fact or "",
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        attributes=dict(edge.attributes or {}),
        episodes=[str(episode) for episode in (edge.episodes or [])],
        created_at=_iso(edge.created_at),
        valid_at=_iso(edge.valid_at),
        invalid_at=_iso(edge.invalid_at),
        expired_at=_iso(edge.expired_at),
    )


def to_episode_view(episode: EpisodicNode) -> EpisodeView:
    return EpisodeView(
        uuid_=episode.uuid,
        content=episode.content or "",
        created_at=_iso(episode.created_at),
        source_description=episode.source_description or "",
    )
