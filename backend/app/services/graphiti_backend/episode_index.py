"""Map episode UUIDs back to the graph they belong to.

The Zep API polls ``graph.episode.get(uuid_=...)`` with no graph context, but
the FalkorDB driver needs to know which graph key to read. graphiti gives each
``group_id`` its own FalkorDB graph, so the adapter records the association
when it creates an episode.
"""

from __future__ import annotations

import json
import os
import threading

from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.episodes")

_INDEX_FILENAME = "episodes.json"


class EpisodeIndex:
    """A persisted ``episode_uuid -> graph_id`` map."""

    def __init__(self, state_dir: str) -> None:
        self._state_dir = state_dir
        self._path = os.path.join(state_dir, _INDEX_FILENAME)
        self._lock = threading.Lock()
        self._entries: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning("Ignoring unreadable episode index %s: %s", self._path, error)
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _persist(self) -> None:
        os.makedirs(self._state_dir, exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._entries, handle)
        os.replace(tmp_path, self._path)

    def record(self, episode_uuid: str, graph_id: str) -> None:
        with self._lock:
            if self._entries.get(episode_uuid) == graph_id:
                return
            self._entries[episode_uuid] = graph_id
            self._persist()

    def graph_of(self, episode_uuid: str) -> str | None:
        with self._lock:
            return self._entries.get(episode_uuid)

    def forget_graph(self, graph_id: str) -> None:
        """Drop every entry for a deleted graph."""

        with self._lock:
            remaining = {
                uuid: gid for uuid, gid in self._entries.items() if gid != graph_id
            }
            if len(remaining) == len(self._entries):
                return
            self._entries = remaining
            self._persist()
