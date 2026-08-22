#!/usr/bin/env python3
"""Run a manual, real validation of the self-hosted Graphiti backend.

This mirrors ``validate_zep_cloud_integration.py`` for ``ZEP_BACKEND=graphiti``:
it drives the same MiroFish call sites (graph create, ontology, Batch API,
episode polling, pagination, search, node detail, delete) against a live
FalkorDB and a live LLM/embedder, and prints a JSON report.

It is intentionally excluded from the automated test suite — it needs real
services. The test graph is deleted unless ``--keep-graph`` is supplied.

    cd backend && uv run python scripts/validate_graphiti_local_integration.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

# The adapter is only reachable when the backend is in graphiti mode; set it
# before app.config is imported so Config picks it up.
os.environ.setdefault("ZEP_BACKEND", "graphiti")

from zep_cloud import BatchAddItem, EntityEdgeSourceTarget, NotFoundError  # noqa: E402
from zep_cloud.external_clients.ontology import (  # noqa: E402
    EdgeModel,
    EntityModel,
    EntityText,
)
from pydantic import Field  # noqa: E402

from app.config import Config  # noqa: E402
from app.services.graphiti_backend import build_graphiti_adapter  # noqa: E402
from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes  # noqa: E402

BATCH_POLL_INTERVAL_SECONDS = 3.0
BATCH_TIMEOUT_SECONDS = 1800.0
TERMINAL_BATCH_STATES = {"succeeded", "partial", "failed"}


@dataclass(frozen=True)
class SourceChunk:
    data: str
    created_at: str


SOURCE_CHUNKS = [
    SourceChunk(
        "澜舟科技（企业稳定标识 LZ-TECH）是一家风电智能运维公司。"
        "公司总部位于星港市，周岚自 2024 年起担任首席执行官（CEO）。",
        "2026-01-05T09:00:00Z",
    ),
    SourceChunk(
        "澜舟科技研发了产品智巡平台（产品稳定标识 ZHIXUN-01）。"
        "智巡平台当前处于试点阶段，用于识别风机叶片异常。",
        "2026-01-12T10:00:00Z",
    ),
    SourceChunk(
        "星港能源集团与澜舟科技在 2026 年 2 月建立了战略合作关系，"
        "由星港能源集团在其风场部署智巡平台。",
        "2026-02-03T14:00:00Z",
    ),
]


class Organization(EntityModel):
    """An organization such as a company or a group."""

    industry: EntityText = Field(description="The industry it operates in", default=None)


class Person(EntityModel):
    """A person mentioned in the source documents."""

    role: EntityText = Field(description="Their professional role", default=None)


class Product(EntityModel):
    """A product or platform built by an organization."""

    stage: EntityText = Field(description="Its current lifecycle stage", default=None)


class WorksFor(EdgeModel):
    """Employment relationship between a person and an organization."""

    since: EntityText = Field(description="When the relationship started", default=None)


class PartnersWith(EdgeModel):
    """Strategic partnership between two organizations."""

    since: EntityText = Field(description="When the partnership started", default=None)


ENTITY_TYPES = {"Organization": Organization, "Person": Person, "Product": Product}
EDGE_TYPES = {
    "WORKS_FOR": (
        WorksFor,
        [EntityEdgeSourceTarget(source="Person", target="Organization")],
    ),
    "PARTNERS_WITH": (
        PartnersWith,
        [EntityEdgeSourceTarget(source="Organization", target="Organization")],
    ),
}


def _wait_for_batch(client: Any, batch_id: str, timeout: float) -> Any:
    deadline = time.time() + timeout
    while True:
        summary = client.batch.get(batch_id=batch_id)
        if summary.status in TERMINAL_BATCH_STATES:
            return summary
        if time.time() > deadline:
            raise TimeoutError(
                f"Batch {batch_id} did not reach a terminal state within {timeout}s "
                f"(last status: {summary.status})"
            )
        print(
            f"  batch {summary.status}: "
            f"{summary.progress.succeeded_items}/{summary.progress.total_items} done",
            file=sys.stderr,
        )
        time.sleep(BATCH_POLL_INTERVAL_SECONDS)


def _node_view(node: Any) -> dict[str, Any]:
    return {
        "uuid": node.uuid_,
        "name": node.name,
        "labels": node.labels,
        "summary": (node.summary or "")[:120],
        "attributes": node.attributes,
    }


def _edge_view(edge: Any) -> dict[str, Any]:
    return {
        "uuid": edge.uuid_,
        "name": edge.name,
        "fact": (edge.fact or "")[:160],
        "source_node_uuid": edge.source_node_uuid,
        "target_node_uuid": edge.target_node_uuid,
        "attributes": edge.attributes,
    }


def run_validation(
    *, keep_graph: bool, page_size: int, result: dict[str, Any]
) -> dict[str, Any]:
    result.update(
        {
        "config": {
            "falkordb_uri": Config.GRAPHITI_FALKORDB_URI,
            "llm_base_url": Config.GRAPHITI_LLM_BASE_URL or Config.LLM_BASE_URL,
            "llm_model": Config.GRAPHITI_LLM_MODEL_NAME or Config.LLM_MODEL_NAME,
            "embedder_base_url": Config.GRAPHITI_EMBEDDER_BASE_URL,
            "embedder_fallback_base_url": Config.GRAPHITI_EMBEDDER_FALLBACK_BASE_URL,
            "embedder_model": Config.GRAPHITI_EMBEDDER_MODEL,
            "embedder_dim": Config.GRAPHITI_EMBEDDER_DIM,
        }
        }
    )

    client = build_graphiti_adapter()
    graph_id = f"mirofish_validate_{uuid.uuid4().hex[:12]}"
    result["graph_id"] = graph_id
    graph_created = False

    try:
        print(f"[1/8] creating graph {graph_id}", file=sys.stderr)
        client.graph.create(
            graph_id=graph_id, name="Graphiti validation", description="temporary"
        )
        graph_created = True
        result["graph"] = {"created": True, "readable": bool(client.graph.get(graph_id))}

        print("[2/8] setting ontology", file=sys.stderr)
        client.graph.set_ontology(
            graph_ids=[graph_id], entities=ENTITY_TYPES, edges=EDGE_TYPES
        )
        ontology_path = os.path.join(
            Config.UPLOAD_FOLDER, "graph_state", f"{graph_id}_ontology.json"
        )
        result["ontology"] = {"persisted": os.path.exists(ontology_path)}

        print("[3/8] submitting batch", file=sys.stderr)
        batch = client.batch.create(
            metadata={"mirofish_operation_id": "validation", "graph_id": graph_id}
        )
        items = [
            BatchAddItem(
                type="graph_episode",
                graph_id=graph_id,
                data=chunk.data,
                data_type="text",
                created_at=chunk.created_at,
                source_description="validation chunk",
                metadata={"chunk_index": index},
            )
            for index, chunk in enumerate(SOURCE_CHUNKS)
        ]
        details = client.batch.add(batch_id=batch.batch_id, items=items)
        episode_uuids = [detail.episode_uuid for detail in details]
        result["batch"] = {
            "batch_id": batch.batch_id,
            "item_count": len(details),
            "sequence_indexes": [detail.sequence_index for detail in details],
        }

        print("[4/8] processing batch (LLM extraction; this takes a while)", file=sys.stderr)
        client.batch.process(batch_id=batch.batch_id)
        summary = _wait_for_batch(client, batch.batch_id, BATCH_TIMEOUT_SECONDS)
        result["batch"]["status"] = summary.status
        result["batch"]["succeeded_items"] = summary.progress.succeeded_items
        result["batch"]["failed_items"] = summary.progress.failed_items
        if summary.status != "succeeded":
            failures = [
                {"sequence_index": item.sequence_index, "error": item.error}
                for item in client.batch.list_items(batch_id=batch.batch_id).items
                if item.status != "succeeded"
            ]
            result["batch"]["failures"] = failures
            raise AssertionError(f"Batch finished as {summary.status}: {failures}")

        print("[5/8] polling episodes", file=sys.stderr)
        result["episodes"] = {
            "count": len(episode_uuids),
            "all_processed": all(
                client.graph.episode.get(uuid_=episode_uuid).processed
                for episode_uuid in episode_uuids
            ),
            "recent_count": len(
                client.graph.episode.get_by_graph_id(graph_id=graph_id, lastn=50)
            ),
        }

        print("[6/8] paginating nodes and edges", file=sys.stderr)
        nodes = fetch_all_nodes(client, graph_id, page_size=page_size)
        edges = fetch_all_edges(client, graph_id, page_size=page_size)
        node_uuids = [node.uuid_ for node in nodes]
        custom_labels = sorted(
            {label for node in nodes for label in (node.labels or []) if label != "Entity"}
        )
        result["graph_contents"] = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "unique_node_uuids": len(set(node_uuids)) == len(node_uuids),
            "custom_labels": custom_labels,
            "custom_edge_names": sorted(
                {edge.name for edge in edges if edge.name in EDGE_TYPES}
            ),
            "nodes": [_node_view(node) for node in nodes[:20]],
        }
        if not nodes:
            raise AssertionError("Extraction produced no entity nodes")

        print("[7/8] searching and reading node detail", file=sys.stderr)
        edge_search = client.graph.search(
            query="澜舟科技的合作关系与负责人", graph_id=graph_id, limit=20, scope="edges"
        )
        node_search = client.graph.search(
            query="风电智能运维相关的公司与人物",
            graph_id=graph_id,
            limit=20,
            scope="nodes",
            reranker="rrf",
        )
        selected_uuid = node_uuids[0]
        node_detail = client.graph.node.get(uuid_=selected_uuid)
        node_edges = client.graph.node.get_edges(node_uuid=selected_uuid)
        result["searches"] = {
            "edge_hits": len(edge_search.edges),
            "node_hits": len(node_search.nodes),
            "sample_facts": [edge.fact[:120] for edge in edge_search.edges[:5]],
        }
        result["node_detail"] = {
            "node": _node_view(node_detail),
            "edge_count": len(node_edges),
            "edges": [_edge_view(edge) for edge in node_edges[:5]],
        }

        result["runtime_assertions"] = {
            "graph_create_and_read": True,
            "ontology_persisted": result["ontology"]["persisted"],
            "batch_reached_succeeded": True,
            "episodes_reported_processed": result["episodes"]["all_processed"],
            "pagination_returned_unique_nodes": result["graph_contents"]["unique_node_uuids"],
            "custom_entity_labels_observed": bool(custom_labels),
            "search_calls_completed": True,
            "node_detail_readable": True,
        }
        return result

    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)[:500]}
        raise
    finally:
        print("[8/8] cleanup", file=sys.stderr)
        if graph_created and not keep_graph:
            try:
                client.graph.delete(graph_id=graph_id)
                try:
                    client.graph.get(graph_id)
                    result["cleanup"] = {"deleted": True, "still_registered": True}
                except NotFoundError:
                    result["cleanup"] = {"deleted": True, "still_registered": False}
            except Exception as cleanup_error:  # noqa: BLE001 - reported, not raised
                result["cleanup"] = {
                    "deleted": False,
                    "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
                }
        else:
            result["cleanup"] = {"deleted": False, "kept_by_request": keep_graph}
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-graph", action="store_true", help="do not delete the validation graph"
    )
    parser.add_argument(
        "--page-size", type=int, default=2, help="pagination page size (exercises cursors)"
    )
    args = parser.parse_args()

    if Config.ZEP_BACKEND != "graphiti":
        print("ZEP_BACKEND must be 'graphiti' to run this validation", file=sys.stderr)
        return 2

    # The report is filled in place so a failure still prints what was reached.
    report: dict[str, Any] = {}
    try:
        run_validation(
            keep_graph=args.keep_graph, page_size=args.page_size, result=report
        )
        return 0
    except Exception:
        return 1
    finally:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
