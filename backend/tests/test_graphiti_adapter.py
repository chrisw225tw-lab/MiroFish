"""Unit tests for the self-hosted Graphiti backend.

These cover the contract the rest of MiroFish depends on — the Zep-shaped
namespaces, ontology persistence, the local batch state machine, and the fake
cursor pagination — without touching FalkorDB or an LLM.
"""

import json
import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field
from zep_cloud import BatchAddItem, EntityEdgeSourceTarget, NotFoundError

from app.services.graphiti_backend import batch_store, namespaces
from app.services.graphiti_backend.batch_store import (
    ITEM_FAILED,
    ITEM_SUCCEEDED,
    STATUS_DRAFT,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_PROCESSING,
    STATUS_SUCCEEDED,
    BatchManager,
)
from app.services.graphiti_backend.engine import (
    _coerce_datetime,
    _cursor_offset,
    _next_cursor,
    _search_config,
    parse_falkordb_uri,
)
from app.services.graphiti_backend.falkor_params import (
    sanitize_query_params,
    sanitize_row,
)
from app.services.graphiti_backend.json_llm_client import (
    normalize_to_schema,
    parse_json_response,
)
from app.services.graphiti_backend.ontology_store import (
    OntologyStore,
    build_ontology_spec,
)
from app.services.graphiti_backend.shapes import EpisodeView, NodeView
from app.utils import zep_paging


# --------------------------------------------------------------------------
# FalkorDB URI parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("redis://127.0.0.1:6380", ("127.0.0.1", 6380, None)),
        ("127.0.0.1:6380", ("127.0.0.1", 6380, None)),
        ("redis://falkordb", ("falkordb", 6379, None)),
        ("redis://user:secret@db.internal:7000", ("db.internal", 7000, "secret")),
    ],
)
def test_falkordb_uri_parsing(uri, expected):
    parsed = parse_falkordb_uri(uri)
    assert (parsed["host"], parsed["port"], parsed["password"]) == expected


def test_falkordb_uri_rejects_hostless_value():
    with pytest.raises(ValueError, match="Invalid FalkorDB URI"):
        parse_falkordb_uri("redis://")


# --------------------------------------------------------------------------
# Ontology conversion and persistence
# --------------------------------------------------------------------------


class _Person(BaseModel):
    """A person mentioned in the source documents."""

    role: str | None = Field(default=None, description="Their role")


class _Works(BaseModel):
    """Employment relationship."""

    since: str | None = Field(default=None, description="Start year")


def test_ontology_spec_maps_entities_edges_and_source_targets():
    spec = build_ontology_spec(
        {"Person": _Person},
        {"WORKS_AT": (_Works, [EntityEdgeSourceTarget(source="Person", target="Org")])},
    )

    assert [entity.name for entity in spec.entities] == ["Person"]
    assert spec.entities[0].attributes == {"role": "Their role"}
    assert spec.entities[0].description == "A person mentioned in the source documents."
    assert spec.edge_type_map == {("Person", "Org"): ["WORKS_AT"]}

    kwargs = spec.as_graphiti_kwargs()
    assert set(kwargs["entity_types"]) == {"Person"}
    assert set(kwargs["edge_types"]) == {"WORKS_AT"}
    assert "since" in kwargs["edge_types"]["WORKS_AT"].model_fields


def test_ontology_spec_drops_fields_reserved_by_graphiti():
    class _Clashing(BaseModel):
        """Type with a reserved attribute."""

        summary: str | None = Field(default=None, description="not allowed")
        detail: str | None = Field(default=None, description="allowed")

    spec = build_ontology_spec({"Thing": _Clashing}, None)

    assert spec.entities[0].attributes == {"detail": "allowed"}


def test_ontology_store_round_trips_through_disk(tmp_path):
    spec = build_ontology_spec(
        {"Person": _Person},
        {"WORKS_AT": (_Works, [EntityEdgeSourceTarget(source="Person", target="Org")])},
    )
    store = OntologyStore(str(tmp_path))
    store.set("graph-1", spec)

    on_disk = json.loads((tmp_path / "graph-1_ontology.json").read_text(encoding="utf-8"))
    assert on_disk["entities"][0]["name"] == "Person"

    reloaded = OntologyStore(str(tmp_path)).get("graph-1")
    assert reloaded is not None
    assert reloaded.edge_type_map == {("Person", "Org"): ["WORKS_AT"]}
    assert reloaded.as_graphiti_kwargs()["entity_types"]["Person"].model_fields.keys() == {
        "role"
    }


def test_ontology_store_returns_no_kwargs_for_unknown_graph(tmp_path):
    assert OntologyStore(str(tmp_path)).episode_kwargs("missing") == {}


def test_ontology_store_delete_removes_state(tmp_path):
    store = OntologyStore(str(tmp_path))
    store.set("graph-1", build_ontology_spec({"Person": _Person}, None))
    store.delete("graph-1")

    assert not os.path.exists(tmp_path / "graph-1_ontology.json")
    assert OntologyStore(str(tmp_path)).get("graph-1") is None


# --------------------------------------------------------------------------
# Batch state machine
# --------------------------------------------------------------------------


def _item(graph_id="graph-1", data="chunk", index=0):
    return BatchAddItem(
        type="graph_episode",
        graph_id=graph_id,
        data=data,
        data_type="text",
        source_description="doc",
        metadata={"chunk_index": index},
    )


def _drain(manager, batch_id, timeout=5.0):
    """Run the batch and wait for its worker thread to settle."""

    manager.process(batch_id)
    worker = manager._workers.get(batch_id)
    if worker is not None:
        worker.join(timeout)
    return manager.get(batch_id)


def test_batch_add_assigns_sequence_indexes_and_episode_uuids(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    batch_id = manager.create({"graph_id": "graph-1"}).batch_id

    details = manager.add(batch_id, [_item(index=0), _item(index=1)])

    assert [item.sequence_index for item in details] == [0, 1]
    assert all(item.episode_uuid for item in details)
    assert len({item.episode_uuid for item in details}) == 2
    assert manager.get(batch_id).status == STATUS_DRAFT


def test_batch_processes_every_item_and_reports_success(tmp_path):
    ingested = []
    manager = BatchManager(str(tmp_path), ingest=ingested.append)
    batch_id = manager.create().batch_id
    manager.add(batch_id, [_item(data="a"), _item(data="b")])

    summary = _drain(manager, batch_id)

    assert [item.data for item in ingested] == ["a", "b"]
    assert summary.status == STATUS_SUCCEEDED
    assert summary.progress.succeeded_items == 2
    assert summary.progress.percent_complete == 100.0


def test_batch_reports_partial_when_some_items_fail(tmp_path):
    def ingest(item):
        if item.data == "bad":
            raise RuntimeError("extraction failed")

    manager = BatchManager(str(tmp_path), ingest=ingest)
    batch_id = manager.create().batch_id
    manager.add(batch_id, [_item(data="ok"), _item(data="bad")])

    summary = _drain(manager, batch_id)

    assert summary.status == STATUS_PARTIAL
    assert summary.progress.succeeded_items == 1
    assert summary.progress.failed_items == 1
    statuses = [item.status for item in manager.list_items(batch_id).items]
    assert statuses == [ITEM_SUCCEEDED, ITEM_FAILED]


def test_batch_reports_failed_when_no_item_succeeds(tmp_path):
    manager = BatchManager(
        str(tmp_path), ingest=lambda item: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    batch_id = manager.create().batch_id
    manager.add(batch_id, [_item()])

    assert _drain(manager, batch_id).status == STATUS_FAILED


def test_batch_add_rejects_items_without_graph_or_data(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    batch_id = manager.create().batch_id

    with pytest.raises(ValueError, match="graph_id and data"):
        manager.add(batch_id, [BatchAddItem(type="graph_episode", data="x")])


def test_batch_list_and_list_items_paginate_with_advancing_cursors(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    batch_id = manager.create().batch_id
    manager.add(batch_id, [_item(index=i) for i in range(5)])

    first = manager.list_items(batch_id, limit=2)
    second = manager.list_items(batch_id, limit=2, cursor=first.next_cursor)
    third = manager.list_items(batch_id, limit=2, cursor=second.next_cursor)

    assert [item.sequence_index for item in first.items] == [0, 1]
    assert [item.sequence_index for item in second.items] == [2, 3]
    assert [item.sequence_index for item in third.items] == [4]
    assert third.next_cursor is None


def test_batch_list_finds_a_batch_by_its_operation_metadata(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    manager.create({"mirofish_operation_id": "op-1", "graph_id": "graph-1"})

    page = manager.list(limit=100)

    assert page.batches[0].metadata["mirofish_operation_id"] == "op-1"
    assert page.next_cursor is None


def test_batch_state_survives_restart_and_settles_interrupted_work(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    batch_id = manager.create({"graph_id": "graph-1"}).batch_id
    manager.add(batch_id, [_item(data="a"), _item(data="b")])
    # Simulate a crash mid-flight: one item landed, the batch is still marked
    # processing, and the worker thread dies with the process.
    manager._batches[batch_id]["items"][0].status = ITEM_SUCCEEDED
    manager._batches[batch_id]["status"] = STATUS_PROCESSING
    manager._persist(batch_id)

    restarted = BatchManager(str(tmp_path), ingest=lambda item: None)
    summary = restarted.get(batch_id)

    assert summary.status == STATUS_PARTIAL
    assert summary.metadata["graph_id"] == "graph-1"
    assert summary.progress.succeeded_items == 1


def test_batch_namespace_translates_unknown_batch_to_zep_not_found(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    namespace = namespaces.BatchNamespace(SimpleNamespace(batches=manager))

    with pytest.raises(NotFoundError):
        namespace.get("does-not-exist")


def test_find_episode_item_locates_a_staged_episode(tmp_path):
    manager = BatchManager(str(tmp_path), ingest=lambda item: None)
    batch_id = manager.create().batch_id
    staged = manager.add(batch_id, [_item()])[0]

    assert manager.find_episode_item(staged.episode_uuid) is staged
    assert manager.find_episode_item("unknown") is None


# --------------------------------------------------------------------------
# Pagination contract with utils/zep_paging
# --------------------------------------------------------------------------


class _FakePagingEngine:
    """Serve SKIP/LIMIT pages the way GraphitiEngine does."""

    def __init__(self, uuids):
        self._uuids = list(uuids)
        self.calls = []

    def _page(self, graph_id, limit, cursor):
        self.calls.append({"limit": limit, "cursor": cursor})
        offset = _cursor_offset(cursor)
        window = self._uuids[offset : offset + limit]
        views = [
            NodeView(uuid_=uuid, name=uuid, labels=["Entity"], summary="", attributes={})
            for uuid in window
        ]
        return views, _next_cursor(views, limit, offset)

    page_nodes = _page
    page_edges = _page


def test_zep_paging_walks_every_page_through_the_adapter():
    engine = _FakePagingEngine(["u1", "u2", "u3", "u4", "u5"])
    client = SimpleNamespace(graph=SimpleNamespace(node=namespaces.NodeNamespace(engine)))

    nodes = zep_paging.fetch_all_nodes(client, "graph-1", page_size=2)

    assert [node.uuid_ for node in nodes] == ["u1", "u2", "u3", "u4", "u5"]
    assert [call["cursor"] for call in engine.calls] == [None, "2", "4"]


def test_zep_paging_stops_when_a_short_page_ends_the_walk():
    engine = _FakePagingEngine(["u1", "u2", "u3"])
    client = SimpleNamespace(graph=SimpleNamespace(edge=namespaces.EdgeNamespace(engine)))

    edges = zep_paging.fetch_all_edges(client, "graph-1", page_size=2)

    assert len(edges) == 3
    assert len(engine.calls) == 2


def test_paging_never_repeats_a_cursor_on_an_exact_multiple_page_count():
    """A full final page must still terminate rather than loop forever."""

    engine = _FakePagingEngine(["u1", "u2", "u3", "u4"])
    client = SimpleNamespace(graph=SimpleNamespace(node=namespaces.NodeNamespace(engine)))

    nodes = zep_paging.fetch_all_nodes(client, "graph-1", page_size=2)

    assert [node.uuid_ for node in nodes] == ["u1", "u2", "u3", "u4"]
    assert [call["cursor"] for call in engine.calls] == [None, "2", "4"]


def test_next_cursor_is_absent_on_a_partial_page():
    views = [NodeView(uuid_="u1", name="", labels=[], summary="", attributes={})]

    assert _next_cursor(views, page_size=2) is None
    assert _next_cursor(views, page_size=1) == "1"
    assert _next_cursor(views, page_size=1, offset=7) == "8"


def test_cursor_offset_tolerates_missing_and_broken_cursors():
    assert _cursor_offset(None) == 0
    assert _cursor_offset("12") == 12
    assert _cursor_offset("-3") == 0
    assert _cursor_offset("not-a-number") == 0


# --------------------------------------------------------------------------
# Graph namespace behaviour
# --------------------------------------------------------------------------


class _RecordingEngine:
    def __init__(self):
        self.added = None
        self.searched = None
        self.batches = None

    def add_episode(self, **kwargs):
        self.added = kwargs
        return EpisodeView(uuid_="ep-1", content=kwargs["data"])

    def search(self, **kwargs):
        self.searched = kwargs
        return SimpleNamespace(edges=[], nodes=[], episodes=[])

    def create_graph(self, graph_id, name, description):
        return SimpleNamespace(graph_id=graph_id, name=name, description=description)

    def set_ontology(self, graph_ids, entities, edges):
        self.ontology = (list(graph_ids), entities, edges)


def test_graph_add_folds_metadata_into_the_source_description():
    engine = _RecordingEngine()
    graph = namespaces.GraphNamespace(engine)

    episode = graph.add(
        graph_id="graph-1",
        type="text",
        data="hello",
        created_at="2026-08-09T10:00:00Z",
        source_description="simulation batch",
        metadata={"platform": "twitter", "activity_count": 3},
    )

    assert episode.uuid_ == "ep-1"
    assert engine.added["graph_id"] == "graph-1"
    assert engine.added["created_at"] == "2026-08-09T10:00:00Z"
    assert "simulation batch" in engine.added["source_description"]
    assert "platform=twitter" in engine.added["source_description"]


def test_graph_add_rejects_non_text_episode_types():
    graph = namespaces.GraphNamespace(_RecordingEngine())

    with pytest.raises(ValueError, match="only ingests text episodes"):
        graph.add(graph_id="graph-1", type="json", data="{}")


def test_graph_search_forwards_scope_and_reranker():
    engine = _RecordingEngine()
    graph = namespaces.GraphNamespace(engine)

    graph.search(query="who leads it", graph_id="graph-1", limit=30, scope="nodes", reranker="rrf")

    assert engine.searched == {
        "graph_id": "graph-1",
        "query": "who leads it",
        "limit": 30,
        "scope": "nodes",
        "reranker": "rrf",
    }


def test_search_config_maps_scope_and_degrades_unknown_rerankers():
    edge_config = _search_config("edges", "rrf", limit=7)
    node_config = _search_config("nodes", "mmr", limit=3)
    fallback = _search_config("edges", "cross_encoder", limit=5)

    assert edge_config.limit == 7
    assert node_config.limit == 3
    assert fallback.limit == 5
    # cross_encoder is not wired up locally; it must land on the RRF recipe.
    assert fallback.edge_config.reranker == _search_config("edges", "rrf", 5).edge_config.reranker


# --------------------------------------------------------------------------
# Timestamp coercion
# --------------------------------------------------------------------------


def test_coerce_datetime_accepts_rfc3339_and_attaches_utc():
    parsed = _coerce_datetime("2026-08-09T10:00:00Z")

    assert (parsed.year, parsed.month, parsed.day, parsed.hour) == (2026, 8, 9, 10)
    assert parsed.tzinfo is not None


def test_coerce_datetime_falls_back_to_now_for_unusable_input():
    assert _coerce_datetime("not-a-timestamp").tzinfo is not None
    assert _coerce_datetime(None).tzinfo is not None


# --------------------------------------------------------------------------
# JSON-tolerant LLM client
# --------------------------------------------------------------------------


class _Attributes(BaseModel):
    industry: str | None = Field(default=None, description="Industry")
    since: str | None = Field(default=None, description="Start")


def test_parse_json_response_accepts_a_markdown_fence():
    assert parse_json_response('```json\n{"ok": true}\n```') == {"ok": True}


def test_parse_json_response_extracts_an_object_wrapped_in_prose():
    raw = 'Here you go:\n{"ok": true}\nHope that helps.'

    assert parse_json_response(raw) == {"ok": True}


def test_parse_json_response_rejects_a_reply_with_no_object():
    with pytest.raises(ValueError, match="not JSON"):
        parse_json_response("I cannot answer that.")

    with pytest.raises(ValueError, match="empty response"):
        parse_json_response("")


def test_normalize_unwraps_an_echoed_json_schema_envelope():
    """FalkorDB rejects nested maps, so the schema echo must be unwrapped."""

    payload = {"properties": {"industry": "wind power", "since": "2024"}}

    assert normalize_to_schema(payload, _Attributes) == {
        "industry": "wind power",
        "since": "2024",
    }


def test_normalize_keeps_a_correctly_shaped_reply_untouched():
    payload = {"industry": "wind power", "since": None}

    assert normalize_to_schema(payload, _Attributes) == payload


def test_normalize_flattens_a_nested_value_into_json_text():
    payload = {"industry": {"value": "wind power", "confidence": 0.9}}

    normalized = normalize_to_schema(payload, _Attributes)

    assert json.loads(normalized["industry"]) == {"value": "wind power", "confidence": 0.9}


def test_normalize_is_a_no_op_without_a_response_model():
    payload = {"properties": {"anything": 1}}

    assert normalize_to_schema(payload, None) == payload


def test_batch_module_exposes_the_statuses_graph_builder_waits_on():
    terminal = {STATUS_SUCCEEDED, STATUS_PARTIAL, STATUS_FAILED}

    assert terminal <= {
        value
        for name, value in vars(batch_store).items()
        if name.startswith("STATUS_")
    }


# --------------------------------------------------------------------------
# FalkorDB parameter sanitizing
# --------------------------------------------------------------------------


def test_sanitizer_passes_through_primitives_and_primitive_arrays():
    row = {
        "uuid": "u1",
        "created_at": 1700000000,
        "score": 0.5,
        "active": True,
        "missing": None,
        "labels": ["Entity", "Person"],
        "name_embedding": [0.1, 0.2, 0.3],
    }

    assert sanitize_row(row) == row


def test_sanitizer_serializes_a_nested_map_property():
    row = {"uuid": "u1", "properties": {"since": None}}

    sanitized = sanitize_row(row)

    assert sanitized["uuid"] == "u1"
    assert json.loads(sanitized["properties"]) == {"since": None}


def test_sanitizer_serializes_an_array_containing_maps():
    row = {"sources": [{"a": 1}, "plain"]}

    assert json.loads(sanitize_row(row)["sources"]) == [{"a": 1}, "plain"]


def test_sanitizer_only_touches_values_inside_unwound_rows():
    params = {
        "entity_edges": [{"uuid": "e1", "attrs": {"x": 1}}],
        "group_ids": ["graph-1"],
        "limit": 100,
    }

    sanitized = sanitize_query_params(params)

    assert sanitized["group_ids"] == ["graph-1"]
    assert sanitized["limit"] == 100
    assert json.loads(sanitized["entity_edges"][0]["attrs"]) == {"x": 1}


def test_normalize_drops_attributes_the_ontology_never_declared():
    payload = {"industry": "wind power", "invented": {"nested": True}}

    assert normalize_to_schema(payload, _Attributes) == {"industry": "wind power"}


class _OpenAttributes(BaseModel):
    model_config = {"extra": "allow"}

    industry: str | None = Field(default=None, description="Industry")


def test_normalize_keeps_extras_when_the_model_allows_them():
    payload = {"industry": "wind power", "extra": "kept"}

    assert normalize_to_schema(payload, _OpenAttributes) == payload
