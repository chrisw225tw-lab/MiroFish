"""Background asyncio loop so Flask's synchronous code can drive graphiti-core.

graphiti-core is async-only while MiroFish's services are synchronous. A single
dedicated loop on a daemon thread keeps one shared FalkorDB connection pool and
lets every adapter call block on ``run()`` without creating a loop per request.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.runtime")

T = TypeVar("T")

# Graph ingestion (LLM extraction + embedding) can be slow; keep the ceiling in
# the same order of magnitude as ZEP_INGESTION_WAIT_TIMEOUT_SECONDS.
DEFAULT_OPERATION_TIMEOUT_SECONDS = 900.0


class AsyncLoopRunner:
    """Own one asyncio loop on a daemon thread and submit coroutines to it."""

    def __init__(self, name: str = "mirofish-graphiti-loop") -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_forever, name=name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("Graphiti async loop failed to start")

    def _run_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run ``coro`` on the background loop and return its result."""

        if self._closed or not self._thread.is_alive():
            coro.close()
            raise RuntimeError("Graphiti async loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(
                timeout if timeout is not None else DEFAULT_OPERATION_TIMEOUT_SECONDS
            )
        except TimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        """Stop the loop. Idempotent; safe to call from any thread."""

        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logger.warning("Graphiti async loop thread did not stop within 10s")
