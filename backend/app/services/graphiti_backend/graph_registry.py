"""Durable bookkeeping for the graphs MiroFish created in FalkorDB.

graphiti has no notion of a "graph" object — it only has ``group_id`` labels on
nodes. Zep does, and MiroFish relies on ``graph.get`` raising for an unknown ID
during create reconciliation, so the adapter keeps its own small registry.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.registry")

_REGISTRY_FILENAME = "graphs.json"


class GraphRegistry:
    """A JSON-file registry of graph IDs and their metadata."""

    def __init__(self, state_dir: str) -> None:
        self._path = os.path.join(state_dir, _REGISTRY_FILENAME)
        self._state_dir = state_dir
        self._lock = threading.Lock()
        self._graphs: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning("Ignoring unreadable graph registry %s: %s", self._path, error)
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _persist(self) -> None:
        os.makedirs(self._state_dir, exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._graphs, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def register(self, graph_id: str, name: str, description: str | None) -> dict[str, Any]:
        with self._lock:
            record = {
                "graph_id": graph_id,
                "name": name,
                "description": description,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._graphs[graph_id] = record
            self._persist()
            return dict(record)

    def get(self, graph_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._graphs.get(graph_id)
            return dict(record) if record else None

    def remove(self, graph_id: str) -> None:
        with self._lock:
            if self._graphs.pop(graph_id, None) is not None:
                self._persist()

    def graph_ids(self) -> list[str]:
        with self._lock:
            return list(self._graphs)
