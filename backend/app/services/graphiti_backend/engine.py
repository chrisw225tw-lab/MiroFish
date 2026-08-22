"""Synchronous graphiti-core operations behind one shared event loop.

Everything in this module is the "verb" layer: it owns the Graphiti instance,
the FalkorDB drivers, the ontology registry, and the batch manager. The
Zep-shaped facade lives in ``namespaces``/``adapter`` and only calls in here.

One graphiti-on-FalkorDB detail shapes the whole design: graphiti stores each
``group_id`` in its own FalkorDB graph key and swaps its driver to reach it.
So the engine keeps one cloned driver per graph and always passes it
explicitly, instead of relying on whichever graph the last write touched.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.edges import get_entity_edge_from_record
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    get_entity_node_from_record,
)
from graphiti_core.search import search_config_recipes as recipes
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from zep_cloud import NotFoundError

from ...utils.logger import get_logger
from .batch_store import BatchItem, BatchManager
from .episode_index import EpisodeIndex
from .falkor_params import install_param_sanitizer
from .graph_registry import GraphRegistry
from .ontology_store import OntologyStore, build_ontology_spec
from .runtime import AsyncLoopRunner
from .shapes import (
    EdgeView,
    EpisodeView,
    GraphSummary,
    NodeView,
    SearchResults,
    to_edge_view,
    to_episode_view,
    to_node_view,
)

logger = get_logger("mirofish.graphiti.engine")

DEFAULT_FALKORDB_PORT = 6379
MAX_PAGE_SIZE = 100

# Zep exposes several rerankers; graphiti's cross-encoder reranker costs one LLM
# call per candidate, which is not worth it behind a subscription proxy. Only
# the embedding-local rerankers are wired up; anything else degrades to RRF.
_EDGE_RECIPES = {
    "rrf": recipes.EDGE_HYBRID_SEARCH_RRF,
    "mmr": recipes.EDGE_HYBRID_SEARCH_MMR,
    "episode_mentions": recipes.EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
}
_NODE_RECIPES = {
    "rrf": recipes.NODE_HYBRID_SEARCH_RRF,
    "mmr": recipes.NODE_HYBRID_SEARCH_MMR,
    "episode_mentions": recipes.NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
}


_NODE_PAGE_QUERY = """
    MATCH (n:Entity)
    WHERE n.group_id IN $group_ids
    RETURN
    {return_clause}
    ORDER BY n.uuid
    SKIP $offset
    LIMIT $limit
"""

_EDGE_PAGE_QUERY = """
    MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
    WHERE e.group_id IN $group_ids
    RETURN
    {return_clause}
    ORDER BY e.uuid
    SKIP $offset
    LIMIT $limit
"""


def parse_falkordb_uri(uri: str) -> dict[str, Any]:
    """Split ``redis://[user:pass@]host:port`` into FalkorDriver arguments."""

    parsed = urlparse(uri if "://" in uri else f"redis://{uri}")
    if not parsed.hostname:
        raise ValueError(f"Invalid FalkorDB URI: {uri}")
    return {
        "host": parsed.hostname,
        "port": parsed.port or DEFAULT_FALKORDB_PORT,
        "username": parsed.username,
        "password": parsed.password,
    }


def _search_config(scope: str, reranker: str | None, limit: int) -> SearchConfig:
    normalized = (reranker or "rrf").strip().lower()
    table = _NODE_RECIPES if scope == "nodes" else _EDGE_RECIPES
    recipe = table.get(normalized)
    if recipe is None:
        logger.debug("Reranker %r is not supported locally; using RRF", reranker)
        recipe = table["rrf"]
    config = recipe.model_copy(deep=True)
    config.limit = limit
    return config


class GraphitiEngine:
    """Own the Graphiti instance and expose blocking operations."""

    def __init__(
        self,
        *,
        falkordb_uri: str,
        llm_client: LLMClient,
        embedder: EmbedderClient,
        state_dir: str,
        cross_encoder: Any = None,
        database: str = "mirofish",
    ) -> None:
        self._runner = AsyncLoopRunner()
        self._driver_kwargs = parse_falkordb_uri(falkordb_uri)
        self._default_database = database
        self._driver = install_param_sanitizer(
            FalkorDriver(database=database, **self._driver_kwargs)
        )
        self._graphiti = Graphiti(
            graph_driver=self._driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
        # add_episode swaps the shared Graphiti driver to the episode's group.
        # Serialize writes so two graphs cannot interleave that swap.
        self._write_lock = asyncio.Lock()
        self._graph_drivers: dict[str, FalkorDriver] = {}

        self.ontology = OntologyStore(state_dir)
        self.registry = GraphRegistry(state_dir)
        self.episodes = EpisodeIndex(state_dir)
        self.batches = BatchManager(state_dir, self._ingest_batch_item)
        logger.info(
            "Graphiti backend ready (falkordb=%s:%s, default db=%s)",
            self._driver_kwargs["host"],
            self._driver_kwargs["port"],
            database,
        )

    # ---------- drivers ----------

    def _driver_for(self, graph_id: str) -> FalkorDriver:
        """Return the driver bound to ``graph_id``'s FalkorDB graph key."""

        driver = self._graph_drivers.get(graph_id)
        if driver is None:
            driver = install_param_sanitizer(self._driver.clone(database=graph_id))
            self._graph_drivers[graph_id] = driver
        return driver

    def _known_graph_ids(self) -> list[str]:
        return self.registry.graph_ids()

    # ---------- lifecycle ----------

    def close(self) -> None:
        try:
            self._runner.run(self._graphiti.close(), timeout=30.0)
        except Exception as error:  # noqa: BLE001 - shutdown must not raise
            logger.warning("Graphiti close failed: %s", error)
        finally:
            self._runner.close()

    # ---------- graphs ----------

    def create_graph(self, graph_id: str, name: str, description: str | None) -> GraphSummary:
        # Indices live per FalkorDB graph key, so they are built per graph.
        self._runner.run(self._driver_for(graph_id).build_indices_and_constraints())
        record = self.registry.register(graph_id, name, description)
        return GraphSummary(
            graph_id=graph_id,
            name=record["name"],
            description=record["description"],
            created_at=record["created_at"],
        )

    def get_graph(self, graph_id: str) -> GraphSummary:
        record = self.registry.get(graph_id)
        if record is None:
            raise NotFoundError(body=f"graph {graph_id} not found")
        return GraphSummary(
            graph_id=graph_id,
            name=record.get("name"),
            description=record.get("description"),
            created_at=record.get("created_at"),
        )

    def delete_graph(self, graph_id: str) -> None:
        driver = self._driver_for(graph_id)
        self._runner.run(clear_data(driver, group_ids=[graph_id]))
        # clear_data empties the graph but leaves the FalkorDB key behind. Each
        # MiroFish graph owns its key, so drop it to complete the delete.
        self._runner.run(_drop_falkordb_graph(driver, graph_id))
        self._graph_drivers.pop(graph_id, None)
        self.ontology.delete(graph_id)
        self.episodes.forget_graph(graph_id)
        self.registry.remove(graph_id)

    def set_ontology(
        self,
        graph_ids: Sequence[str],
        entities: Mapping[str, Any] | None,
        edges: Mapping[str, Any] | None,
    ) -> None:
        spec = build_ontology_spec(entities, edges)
        for graph_id in graph_ids:
            self.ontology.set(graph_id, spec)

    # ---------- episodes ----------

    def add_episode(
        self,
        *,
        graph_id: str,
        data: str,
        source_description: str = "",
        created_at: str | datetime | None = None,
        name: str | None = None,
    ) -> EpisodeView:
        reference_time = _coerce_datetime(created_at)
        episode_name = name or f"{graph_id}-episode"
        results = self._runner.run(
            self._add_episode_locked(
                graph_id=graph_id,
                name=episode_name,
                data=data,
                source_description=source_description or "MiroFish episode",
                reference_time=reference_time,
            )
        )
        view = to_episode_view(results.episode)
        self.episodes.record(view.uuid_, graph_id)
        return view

    async def _add_episode_locked(
        self,
        *,
        graph_id: str,
        name: str,
        data: str,
        source_description: str,
        reference_time: datetime,
    ) -> Any:
        async with self._write_lock:
            return await self._graphiti.add_episode(
                name=name,
                episode_body=data,
                source_description=source_description,
                reference_time=reference_time,
                source=EpisodeType.text,
                group_id=graph_id,
                **self.ontology.episode_kwargs(graph_id),
            )

    def get_episode(self, episode_uuid: str) -> EpisodeView:
        """Resolve an episode handle to its current ingestion state.

        Batch handles are answered from the batch store: they are issued at
        ``batch.add`` time, before graphiti has created anything, so the caller
        can poll them the same way it polls Zep's episode UUIDs.
        """

        staged = self.batches.find_episode_item(episode_uuid)
        if staged is not None:
            return EpisodeView(
                uuid_=episode_uuid,
                content=staged.data,
                created_at=staged.created_at,
                source_description=staged.source_description,
                processed=staged.status == "succeeded",
            )

        graph_id = self.episodes.graph_of(episode_uuid)
        candidates = [graph_id] if graph_id else self._known_graph_ids()
        for candidate in candidates:
            try:
                episode = self._runner.run(
                    EpisodicNode.get_by_uuid(self._driver_for(candidate), episode_uuid)
                )
            except NodeNotFoundError:
                continue
            return to_episode_view(episode)
        raise NotFoundError(body=f"episode {episode_uuid} not found")

    def recent_episodes(self, graph_id: str, limit: int) -> list[EpisodeView]:
        episodes = self._runner.run(
            EpisodicNode.get_by_group_ids(
                self._driver_for(graph_id), [graph_id], limit=limit
            )
        )
        return [to_episode_view(episode) for episode in episodes]

    def _ingest_batch_item(self, item: BatchItem) -> None:
        view = self.add_episode(
            graph_id=item.graph_id,
            data=item.data,
            source_description=item.source_description,
            created_at=item.created_at,
            name=f"{item.graph_id}-chunk-{item.sequence_index}",
        )
        item.graphiti_episode_uuid = view.uuid_

    # ---------- nodes and edges ----------

    def get_node(self, node_uuid: str) -> NodeView:
        """Look a node up. Zep's endpoint takes no graph, so graphs are tried."""

        for graph_id in self._known_graph_ids():
            try:
                node = self._runner.run(
                    EntityNode.get_by_uuid(self._driver_for(graph_id), node_uuid)
                )
            except NodeNotFoundError:
                continue
            return to_node_view(node)
        raise NotFoundError(body=f"node {node_uuid} not found")

    def get_node_edges(self, node_uuid: str) -> list[EdgeView]:
        for graph_id in self._known_graph_ids():
            try:
                edges = self._runner.run(
                    EntityEdge.get_by_node_uuid(self._driver_for(graph_id), node_uuid)
                )
            except EdgeNotFoundError:
                continue
            if edges:
                return [to_edge_view(edge) for edge in edges]
        return []

    def page_nodes(
        self, graph_id: str, limit: int, cursor: str | None
    ) -> tuple[list[NodeView], str | None]:
        page_size = _clamp_page_size(limit)
        offset = _cursor_offset(cursor)
        provider = self._driver_for(graph_id).provider
        query = _NODE_PAGE_QUERY.format(
            return_clause=get_entity_node_return_query(provider)
        )
        records = self._runner.run(
            self._paged_query(graph_id, query, page_size, offset)
        )
        views = [to_node_view(get_entity_node_from_record(r, provider)) for r in records]
        return views, _next_cursor(views, page_size, offset)

    def page_edges(
        self, graph_id: str, limit: int, cursor: str | None
    ) -> tuple[list[EdgeView], str | None]:
        page_size = _clamp_page_size(limit)
        offset = _cursor_offset(cursor)
        provider = self._driver_for(graph_id).provider
        query = _EDGE_PAGE_QUERY.format(
            return_clause=get_entity_edge_return_query(provider)
        )
        records = self._runner.run(
            self._paged_query(graph_id, query, page_size, offset)
        )
        views = [to_edge_view(get_entity_edge_from_record(r, provider)) for r in records]
        return views, _next_cursor(views, page_size, offset)

    async def _paged_query(
        self, graph_id: str, query: str, limit: int, offset: int
    ) -> list[Any]:
        """Page with SKIP/LIMIT.

        graphiti's own ``uuid_cursor`` argument is not honoured by the FalkorDB
        driver — the filter never reaches the query, so every page returns the
        same rows. A deterministic ORDER BY plus SKIP/LIMIT is reliable here.
        """

        records, _, _ = await self._driver_for(graph_id).execute_query(
            query,
            group_ids=[graph_id],
            limit=limit,
            offset=offset,
            routing_="r",
        )
        return list(records or [])

    # ---------- search ----------

    def search(
        self,
        *,
        graph_id: str,
        query: str,
        limit: int,
        scope: str = "edges",
        reranker: str | None = None,
    ) -> SearchResults:
        normalized_scope = (scope or "edges").strip().lower()
        config = _search_config(normalized_scope, reranker, limit)
        results = self._runner.run(
            self._graphiti.search_(
                query,
                config=config,
                group_ids=[graph_id],
                driver=self._driver_for(graph_id),
            )
        )
        return SearchResults(
            edges=[to_edge_view(edge) for edge in results.edges],
            nodes=[to_node_view(node) for node in results.nodes],
            episodes=[to_episode_view(episode) for episode in results.episodes],
        )


def _clamp_page_size(limit: Any) -> int:
    try:
        page_size = int(limit)
    except (TypeError, ValueError):
        page_size = MAX_PAGE_SIZE
    return max(1, min(page_size, MAX_PAGE_SIZE))


async def _drop_falkordb_graph(driver: FalkorDriver, graph_id: str) -> None:
    """Remove the FalkorDB graph key itself, tolerating an already-gone key."""

    try:
        await driver.client.select_graph(graph_id).delete()
    except Exception as error:  # noqa: BLE001 - best effort after clear_data
        logger.info("Could not drop FalkorDB graph key %s: %s", graph_id, error)


def _cursor_offset(cursor: str | None) -> int:
    """Read a pagination cursor as a row offset, tolerating a missing value."""

    if cursor is None:
        return 0
    try:
        return max(0, int(cursor))
    except (TypeError, ValueError):
        logger.warning("Unusable pagination cursor %r; restarting from 0", cursor)
        return 0


def _next_cursor(views: Sequence[Any], page_size: int, offset: int = 0) -> str | None:
    """Return the cursor for the next page, or None once the page is short."""

    if len(views) < page_size:
        return None
    return str(offset + len(views))


def _coerce_datetime(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            logger.warning("Unparseable episode timestamp %r; using now()", value)
            return datetime.now(timezone.utc)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)
