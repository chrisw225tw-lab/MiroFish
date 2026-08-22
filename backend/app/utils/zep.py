"""Shared graph client, request limits, and retry policy.

``ZEP_BACKEND`` selects the implementation: ``cloud`` returns the Zep Cloud SDK
client (upstream behaviour), ``graphiti`` returns a locally hosted adapter that
exposes the same namespaces. Everything else in this module is backend-neutral.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

import httpx
from zep_cloud.client import Zep
from zep_cloud.core.api_error import ApiError as ZepApiError

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.zep")

T = TypeVar("T")

ZEP_CLOUD_BASE_URL = "https://api.getzep.com/api/v2"
# Keep request behavior aligned with the zep-cloud 3.25.0 SDK default that
# MiroFish used before introducing the shared client. This is an internal
# integration policy, not a deployment setting users need to tune.
ZEP_HTTP_REQUEST_TIMEOUT_SECONDS = 60.0
# Zep ingestion is asynchronous and may take several minutes. Preserve the
# original GraphBuilder deadline while keeping it separate from HTTP timeout.
ZEP_INGESTION_WAIT_TIMEOUT_SECONDS = 600
MAX_ZEP_SEARCH_QUERY_CHARS = 400
MAX_ZEP_SEARCH_RESULTS = 50
# Transport failures raised by the Graphiti backend's dependencies. Matched by
# name so this module keeps working without redis/openai installed.
_RETRYABLE_FOREIGN_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "BusyLoadingError",
        "ConnectionError",
        "TimeoutError",
    }
)


def normalize_zep_search_query(query: Any) -> str:
    """Return a non-empty query within Zep Cloud's endpoint limit."""

    if not isinstance(query, str):
        raise ValueError("Zep search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Zep search query must not be empty")
    return normalized[:MAX_ZEP_SEARCH_QUERY_CHARS]


def normalize_zep_search_limit(limit: Any) -> int:
    """Clamp a search result limit to the current Zep Cloud contract."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Zep search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Zep search limit must be at least 1")
    return min(normalized, MAX_ZEP_SEARCH_RESULTS)


@lru_cache(maxsize=4)
def _cached_zep_client(api_key: str, timeout: float) -> Zep:
    return Zep(
        api_key=api_key,
        base_url=ZEP_CLOUD_BASE_URL,
        timeout=timeout,
    )


def is_cloud_backend() -> bool:
    """Return whether the configured graph backend is Zep Cloud."""

    return Config.ZEP_BACKEND != "graphiti"


def graph_backend_credentials_missing(api_key: str | None = None) -> bool:
    """Return whether the active backend is missing its credentials.

    Only Zep Cloud needs an API key; the self-hosted backend authenticates
    against FalkorDB and the LLM/embedder endpoints instead, which
    ``Config.validate()`` already checks at startup.
    """

    if not is_cloud_backend():
        return False
    return not (api_key or Config.ZEP_API_KEY or "").strip()


def require_graph_backend_credentials(api_key: str | None = None) -> None:
    """Raise when the active backend cannot be reached with what is configured."""

    if graph_backend_credentials_missing(api_key):
        raise ValueError("ZEP_API_KEY 未配置")


def get_zep_client(api_key: str | None = None, timeout: float | None = None) -> Any:
    """Return the process-shared graph client for the configured backend."""

    if Config.ZEP_BACKEND == "graphiti":
        # Imported lazily so a Zep Cloud deployment never needs graphiti-core.
        from ..services.graphiti_backend import get_graphiti_adapter

        return get_graphiti_adapter()

    return get_zep_cloud_client(api_key=api_key, timeout=timeout)


def get_zep_cloud_client(
    api_key: str | None = None, timeout: float | None = None
) -> Zep:
    """Return a process-shared, explicitly configured Zep Cloud client."""

    # zep-cloud gives ZEP_API_URL precedence even when base_url is explicit.
    # Reject it so this Cloud-only integration cannot silently target a
    # self-hosted or compatibility endpoint.
    if os.environ.get("ZEP_API_URL"):
        raise ValueError("ZEP_API_URL is unsupported; unset it to use Zep Cloud")

    normalized_key = (api_key or Config.ZEP_API_KEY or "").strip()
    if not normalized_key:
        raise ValueError("ZEP_API_KEY 未配置")

    request_timeout = float(
        timeout if timeout is not None else ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
    )
    if request_timeout <= 0:
        raise ValueError("Zep request timeout must be greater than 0")
    return _cached_zep_client(normalized_key, request_timeout)


def clear_zep_client_cache() -> None:
    """Clear cached clients. Intended for tests and controlled reconfiguration."""

    _cached_zep_client.cache_clear()
    if Config.ZEP_BACKEND == "graphiti":
        from ..services.graphiti_backend import clear_graphiti_adapter_cache

        clear_graphiti_adapter_cache()


def is_retryable_zep_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry."""

    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(error, ZepApiError):
        status_code = error.status_code
        return status_code in {408, 429} or (
            status_code is not None and 500 <= status_code <= 599
        )
    # Graphiti backend: FalkorDB (redis-py) and the OpenAI-compatible LLM/embedder
    # raise their own transport errors that are equally safe to retry on reads.
    if type(error).__name__ in _RETRYABLE_FOREIGN_ERROR_NAMES:
        return True
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 429} or 500 <= status_code <= 599
    return False


def _retry_after_seconds(error: BaseException) -> float | None:
    if not isinstance(error, ZepApiError) or not error.headers:
        return None
    value = next(
        (
            header_value
            for header_name, header_value in error.headers.items()
            if header_name.lower() == "retry-after"
        ),
        None,
    )
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def call_zep_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe Zep read only for transport, 408, 429, or 5xx errors."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_zep_error(error):
                raise

            retry_after = _retry_after_seconds(error)
            delay = min(
                retry_after if retry_after is not None else initial_delay * (2 ** (attempt - 1)),
                max_delay,
            )
            logger.warning(
                "Zep %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")
