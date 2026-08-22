"""Zep-client-shaped namespaces over :class:`GraphitiEngine`.

Method names, keyword names, and return attributes deliberately mirror
``zep_cloud.client.Zep`` so MiroFish's services work against either backend
without a single call-site change.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from zep_cloud import NotFoundError

from .batch_store import BatchNotFoundError
from .engine import GraphitiEngine
from .shapes import EpisodeView, RawResponse

NEXT_CURSOR_HEADER = "zep-next-cursor"


class _RawNodeResponse:
    """``client.graph.node.with_raw_response`` — data plus cursor headers."""

    def __init__(self, engine: GraphitiEngine, kind: str) -> None:
        self._engine = engine
        self._kind = kind

    def get_by_graph_id(
        self, graph_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> RawResponse:
        pager = self._engine.page_nodes if self._kind == "nodes" else self._engine.page_edges
        items, next_cursor = pager(graph_id, limit, cursor)
        headers = {NEXT_CURSOR_HEADER: next_cursor} if next_cursor else {}
        return RawResponse(data=items, headers=headers)


class NodeNamespace:
    """``client.graph.node``."""

    def __init__(self, engine: GraphitiEngine) -> None:
        self._engine = engine
        self.with_raw_response = _RawNodeResponse(engine, "nodes")

    def get(self, uuid_: str) -> Any:
        return self._engine.get_node(uuid_)

    def get_edges(self, node_uuid: str) -> list[Any]:
        return self._engine.get_node_edges(node_uuid)


class EdgeNamespace:
    """``client.graph.edge``."""

    def __init__(self, engine: GraphitiEngine) -> None:
        self._engine = engine
        self.with_raw_response = _RawNodeResponse(engine, "edges")

    def get(self, uuid_: str) -> Any:
        for edge in self._engine.get_node_edges(uuid_):
            if edge.uuid_ == uuid_:
                return edge
        raise NotFoundError(body=f"edge {uuid_} not found")


class EpisodeNamespace:
    """``client.graph.episode``."""

    def __init__(self, engine: GraphitiEngine) -> None:
        self._engine = engine

    def get(self, uuid_: str) -> EpisodeView:
        return self._engine.get_episode(uuid_)

    def get_by_graph_id(self, graph_id: str, lastn: int = 50) -> list[EpisodeView]:
        return self._engine.recent_episodes(graph_id, lastn)


class GraphNamespace:
    """``client.graph``."""

    def __init__(self, engine: GraphitiEngine) -> None:
        self._engine = engine
        self.node = NodeNamespace(engine)
        self.edge = EdgeNamespace(engine)
        self.episode = EpisodeNamespace(engine)

    def create(
        self, graph_id: str, name: str = "", description: str | None = None
    ) -> Any:
        return self._engine.create_graph(graph_id, name, description)

    def get(self, graph_id: str) -> Any:
        return self._engine.get_graph(graph_id)

    def delete(self, graph_id: str) -> None:
        self._engine.delete_graph(graph_id)

    def set_ontology(
        self,
        graph_ids: Sequence[str],
        entities: Mapping[str, Any] | None = None,
        edges: Mapping[str, Any] | None = None,
    ) -> None:
        self._engine.set_ontology(graph_ids, entities, edges)

    def add(
        self,
        graph_id: str,
        data: str,
        type: str = "text",  # noqa: A002 - Zep's keyword name
        created_at: str | None = None,
        source_description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeView:
        if type != "text":
            raise ValueError(
                f"The Graphiti backend only ingests text episodes, got {type!r}"
            )
        # Zep stores episode metadata server-side; graphiti has no equivalent
        # field, so it is folded into the source description for traceability.
        description = source_description or "MiroFish episode"
        if metadata:
            description = f"{description} | {_render_metadata(metadata)}"
        return self._engine.add_episode(
            graph_id=graph_id,
            data=data,
            source_description=description,
            created_at=created_at,
        )

    def search(
        self,
        query: str,
        graph_id: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str | None = None,
    ) -> Any:
        return self._engine.search(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope=scope,
            reranker=reranker,
        )


class BatchNamespace:
    """``client.batch``."""

    def __init__(self, engine: GraphitiEngine) -> None:
        self._batches = engine.batches

    def create(self, metadata: Mapping[str, Any] | None = None) -> Any:
        return self._batches.create(metadata)

    def add(self, batch_id: str, items: Iterable[Any]) -> list[Any]:
        with _translate_missing_batch(batch_id):
            return self._batches.add(batch_id, items)

    def process(self, batch_id: str) -> Any:
        with _translate_missing_batch(batch_id):
            return self._batches.process(batch_id)

    def get(self, batch_id: str) -> Any:
        with _translate_missing_batch(batch_id):
            return self._batches.get(batch_id)

    def list(self, limit: int = 100, cursor: int | None = None) -> Any:
        return self._batches.list(limit=limit, cursor=cursor)

    def list_items(
        self, batch_id: str, limit: int = 100, cursor: int | None = None
    ) -> Any:
        with _translate_missing_batch(batch_id):
            return self._batches.list_items(batch_id, limit=limit, cursor=cursor)


class _translate_missing_batch:
    """Re-raise an unknown batch ID as the Zep error callers already handle."""

    def __init__(self, batch_id: str) -> None:
        self._batch_id = batch_id

    def __enter__(self) -> "_translate_missing_batch":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None and issubclass(exc_type, BatchNotFoundError):
            raise NotFoundError(body=f"batch {self._batch_id} not found") from exc
        return False


def _render_metadata(metadata: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(metadata.items()))
