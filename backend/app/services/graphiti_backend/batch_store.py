"""Local stand-in for Zep's Batch API, backed by graphiti ingestion.

Zep batches are server-side objects with a durable identity that MiroFish
persists and reconciles after a crash. The local manager keeps that contract:
identities and per-item statuses are written to disk, so a restarted backend
can still answer ``batch.get`` for a batch it started before the restart.
"""

from __future__ import annotations

import json
import os
import threading
import uuid as uuid_module
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.batch")

STATUS_DRAFT = "draft"
STATUS_PROCESSING = "processing"
STATUS_SUCCEEDED = "succeeded"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

ITEM_PENDING = "pending"
ITEM_SUCCEEDED = "succeeded"
ITEM_FAILED = "failed"

# One ingest callable per item: (graph_id, data, source_description, metadata,
# episode_uuid, sequence_index) -> None. Raising marks the item failed.
IngestCallable = Callable[["BatchItem"], None]


@dataclass
class BatchItem:
    """One queued episode inside a batch."""

    sequence_index: int
    graph_id: str
    data: str
    episode_uuid: str
    source_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    status: str = ITEM_PENDING
    error: str | None = None
    # UUID graphiti assigned once the episode actually landed. ``episode_uuid``
    # stays the caller-facing handle, issued before ingestion starts.
    graphiti_episode_uuid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatchItem":
        return cls(**{key: payload.get(key) for key in cls.__dataclass_fields__})


@dataclass
class BatchProgress:
    """Zep ``batch.get().progress`` surface."""

    total_items: int
    succeeded_items: int
    failed_items: int
    pending_items: int
    percent_complete: float


@dataclass
class BatchSummaryView:
    """Zep ``batch.get()`` / ``batch.list()`` surface."""

    batch_id: str
    status: str
    metadata: dict[str, Any]
    progress: BatchProgress


@dataclass
class BatchListPage:
    batches: list[BatchSummaryView]
    next_cursor: int | None


@dataclass
class BatchItemPage:
    items: list[BatchItem]
    next_cursor: int | None


class BatchNotFoundError(KeyError):
    """Raised when a batch identity is unknown to this backend."""


class BatchManager:
    """Track batch identities, queue their items, and ingest them in order."""

    def __init__(self, state_dir: str, ingest: IngestCallable) -> None:
        self._state_dir = state_dir
        self._ingest = ingest
        self._lock = threading.RLock()
        self._batches: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._load_all()

    # ---------- persistence ----------

    def _path(self, batch_id: str) -> str:
        return os.path.join(self._state_dir, f"batch_{batch_id}.json")

    def _load_all(self) -> None:
        if not os.path.isdir(self._state_dir):
            return
        for filename in sorted(os.listdir(self._state_dir)):
            if not (filename.startswith("batch_") and filename.endswith(".json")):
                continue
            path = os.path.join(self._state_dir, filename)
            try:
                with open(path, encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, ValueError) as error:
                logger.warning("Ignoring unreadable batch state %s: %s", path, error)
                continue
            record["items"] = [BatchItem.from_dict(item) for item in record.get("items", [])]
            if record.get("status") == STATUS_PROCESSING:
                # The worker thread died with the previous process. Settle the
                # batch from what actually landed so callers see a terminal state.
                record["status"] = self._terminal_status(record["items"])
            self._batches[record["batch_id"]] = record

    def _persist(self, batch_id: str) -> None:
        record = self._batches[batch_id]
        os.makedirs(self._state_dir, exist_ok=True)
        payload = {
            "batch_id": record["batch_id"],
            "status": record["status"],
            "metadata": record["metadata"],
            "items": [item.to_dict() for item in record["items"]],
        }
        path = self._path(batch_id)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, path)

    # ---------- Zep-compatible surface ----------

    def create(self, metadata: Mapping[str, Any] | None = None) -> BatchSummaryView:
        batch_id = uuid_module.uuid4().hex
        with self._lock:
            self._batches[batch_id] = {
                "batch_id": batch_id,
                "status": STATUS_DRAFT,
                "metadata": dict(metadata or {}),
                "items": [],
            }
            self._persist(batch_id)
            return self._summary(batch_id)

    def add(self, batch_id: str, items: Iterable[Any]) -> list[BatchItem]:
        """Stage episodes and return their details, mirroring ``batch.add``."""

        with self._lock:
            record = self._require(batch_id)
            if record["status"] != STATUS_DRAFT:
                raise ValueError(f"Batch {batch_id} is no longer a draft")

            added: list[BatchItem] = []
            for item in items:
                graph_id = getattr(item, "graph_id", None)
                data = getattr(item, "data", None) or getattr(item, "content", None)
                if not graph_id or not data:
                    raise ValueError("Batch items require graph_id and data")
                batch_item = BatchItem(
                    sequence_index=len(record["items"]),
                    graph_id=graph_id,
                    data=data,
                    episode_uuid=str(uuid_module.uuid4()),
                    source_description=getattr(item, "source_description", "") or "",
                    metadata=dict(getattr(item, "metadata", None) or {}),
                    created_at=getattr(item, "created_at", None),
                )
                record["items"].append(batch_item)
                added.append(batch_item)

            self._persist(batch_id)
            return added

    def process(self, batch_id: str) -> BatchSummaryView:
        """Start ingesting a draft batch on a worker thread."""

        with self._lock:
            record = self._require(batch_id)
            if record["status"] != STATUS_DRAFT:
                return self._summary(batch_id)
            record["status"] = STATUS_PROCESSING
            self._persist(batch_id)
            worker = threading.Thread(
                target=self._run_batch,
                args=(batch_id,),
                name=f"mirofish-graphiti-batch-{batch_id[:8]}",
                daemon=True,
            )
            self._workers[batch_id] = worker
            worker.start()
            return self._summary(batch_id)

    def get(self, batch_id: str) -> BatchSummaryView:
        with self._lock:
            self._require(batch_id)
            return self._summary(batch_id)

    def list(self, limit: int = 100, cursor: int | None = None) -> BatchListPage:
        offset = int(cursor or 0)
        with self._lock:
            batch_ids = list(self._batches)
            page_ids = batch_ids[offset : offset + limit]
            summaries = [self._summary(batch_id) for batch_id in page_ids]
        next_offset = offset + len(page_ids)
        return BatchListPage(
            batches=summaries,
            next_cursor=next_offset if next_offset < len(batch_ids) else None,
        )

    def list_items(
        self, batch_id: str, limit: int = 100, cursor: int | None = None
    ) -> BatchItemPage:
        offset = int(cursor or 0)
        with self._lock:
            record = self._require(batch_id)
            items = list(record["items"])
        page = items[offset : offset + limit]
        next_offset = offset + len(page)
        return BatchItemPage(
            items=page,
            next_cursor=next_offset if next_offset < len(items) else None,
        )

    def find_episode_item(self, episode_uuid: str) -> BatchItem | None:
        """Return the staged item owning ``episode_uuid``, if any."""

        with self._lock:
            for record in self._batches.values():
                for item in record["items"]:
                    if item.episode_uuid == episode_uuid:
                        return item
        return None

    # ---------- internals ----------

    def _require(self, batch_id: str) -> dict[str, Any]:
        record = self._batches.get(batch_id)
        if record is None:
            raise BatchNotFoundError(batch_id)
        return record

    @staticmethod
    def _terminal_status(items: list[BatchItem]) -> str:
        succeeded = sum(1 for item in items if item.status == ITEM_SUCCEEDED)
        failed = sum(1 for item in items if item.status != ITEM_SUCCEEDED)
        if failed == 0:
            return STATUS_SUCCEEDED
        if succeeded == 0:
            return STATUS_FAILED
        return STATUS_PARTIAL

    def _summary(self, batch_id: str) -> BatchSummaryView:
        record = self._batches[batch_id]
        items: list[BatchItem] = record["items"]
        total = len(items)
        succeeded = sum(1 for item in items if item.status == ITEM_SUCCEEDED)
        failed = sum(1 for item in items if item.status == ITEM_FAILED)
        pending = total - succeeded - failed
        return BatchSummaryView(
            batch_id=batch_id,
            status=record["status"],
            metadata=dict(record["metadata"]),
            progress=BatchProgress(
                total_items=total,
                succeeded_items=succeeded,
                failed_items=failed,
                pending_items=pending,
                percent_complete=(100.0 * (succeeded + failed) / total) if total else 100.0,
            ),
        )

    def _run_batch(self, batch_id: str) -> None:
        with self._lock:
            items = list(self._batches[batch_id]["items"])

        for item in items:
            if item.status == ITEM_SUCCEEDED:
                continue
            try:
                self._ingest(item)
                status, error = ITEM_SUCCEEDED, None
            except Exception as ingest_error:  # noqa: BLE001 - recorded per item
                status, error = ITEM_FAILED, str(ingest_error)
                logger.error(
                    "Graphiti batch %s item %s failed: %s",
                    batch_id,
                    item.sequence_index,
                    ingest_error,
                )
            with self._lock:
                item.status = status
                item.error = error
                self._persist(batch_id)

        with self._lock:
            record = self._batches[batch_id]
            record["status"] = self._terminal_status(record["items"])
            self._persist(batch_id)
            self._workers.pop(batch_id, None)
        logger.info("Graphiti batch %s finished with status %s", batch_id, record["status"])
