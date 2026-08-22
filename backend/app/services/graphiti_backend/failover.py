"""Failover wrappers for the graphiti LLM and embedder clients.

Both wrappers share one policy: a transport/5xx/429 failure moves to the next
provider and parks the failed one for a cooldown window, while a 4xx business
error propagates immediately (retrying it on another model would only burn
quota on the same bad request).
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Sequence

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel

from ...utils.logger import get_logger
from .json_llm_client import JsonTolerantOpenAIClient

logger = get_logger("mirofish.graphiti.failover")

DEFAULT_COOLDOWN_SECONDS = 60.0


def is_failover_worthy(error: BaseException) -> bool:
    """Return whether ``error`` is a provider fault worth retrying elsewhere."""

    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        status = getattr(error, "status_code", None)
        return status in {408, 429} or (status is not None and status >= 500)
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


class CooldownRegistry:
    """Track which providers are temporarily parked after a failure."""

    def __init__(self, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def park(self, label: str) -> None:
        with self._lock:
            self._until[label] = time.time() + self._cooldown_seconds

    def is_parked(self, label: str) -> bool:
        with self._lock:
            until = self._until.get(label)
            if until is None:
                return False
            if time.time() >= until:
                del self._until[label]
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            self._until.clear()


class FailoverLLMClient(LLMClient):
    """Try each configured model in order, parking providers that fault."""

    def __init__(
        self,
        clients: Sequence[tuple[str, LLMClient]],
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        if not clients:
            raise ValueError("FailoverLLMClient requires at least one client")
        primary_config = getattr(clients[0][1], "config", None) or LLMConfig()
        super().__init__(config=primary_config, cache=False)
        self._clients = list(clients)
        self._cooldowns = CooldownRegistry(cooldown_seconds)

    @property
    def model_chain(self) -> list[str]:
        return [label for label, _ in self._clients]

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
        last_error: BaseException | None = None
        attempted = False

        for label, client in self._clients:
            if self._cooldowns.is_parked(label):
                logger.info("Graphiti LLM %s is in cooldown; skipping", label)
                continue
            attempted = True
            try:
                # LLMClient.generate_response mutates message content (schema
                # hint, language instruction), so each attempt needs its own copy.
                return await client.generate_response(
                    copy.deepcopy(messages),
                    response_model=response_model,
                    max_tokens=max_tokens,
                    model_size=model_size,
                    group_id=group_id,
                    prompt_name=prompt_name,
                )
            except Exception as error:
                if not is_failover_worthy(error):
                    raise
                last_error = error
                self._cooldowns.park(label)
                logger.warning(
                    "Graphiti LLM %s failed (%s: %s); switching to the next model",
                    label,
                    type(error).__name__,
                    error,
                )

        if last_error is not None:
            raise last_error
        if not attempted:
            raise RuntimeError("All Graphiti LLM models are in cooldown")
        raise AssertionError("unreachable")

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        # Required by LLMClient's ABC. Failover happens in generate_response,
        # which is the entry point graphiti-core actually calls.
        return await self.generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )


class FailoverEmbedder(EmbedderClient):
    """Use a primary embedder endpoint, falling back to a secondary one.

    Both endpoints must serve the same model and dimension: stored vectors are
    only comparable when they come from the same embedding space.
    """

    def __init__(
        self,
        embedders: Sequence[tuple[str, EmbedderClient]],
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        if not embedders:
            raise ValueError("FailoverEmbedder requires at least one embedder")
        self._embedders = list(embedders)
        self._cooldowns = CooldownRegistry(cooldown_seconds)

    @property
    def endpoint_chain(self) -> list[str]:
        return [label for label, _ in self._embedders]

    async def _call(self, method_name: str, payload: Any) -> Any:
        last_error: BaseException | None = None
        attempted = False

        for label, embedder in self._embedders:
            if self._cooldowns.is_parked(label):
                logger.info("Graphiti embedder %s is in cooldown; skipping", label)
                continue
            attempted = True
            try:
                return await getattr(embedder, method_name)(payload)
            except Exception as error:
                if not is_failover_worthy(error):
                    raise
                last_error = error
                self._cooldowns.park(label)
                logger.warning(
                    "Graphiti embedder %s failed (%s: %s); switching endpoint",
                    label,
                    type(error).__name__,
                    error,
                )

        if last_error is not None:
            raise last_error
        if not attempted:
            raise RuntimeError("All Graphiti embedder endpoints are in cooldown")
        raise AssertionError("unreachable")

    async def create(self, input_data: Any) -> list[float]:
        return await self._call("create", input_data)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._call("create_batch", input_data_list)


def build_llm_client(
    *,
    api_key: str,
    base_url: str,
    models: Sequence[str],
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
) -> FailoverLLMClient:
    """Build a failover LLM chain over one OpenAI-compatible endpoint."""

    unique_models = list(dict.fromkeys(model for model in models if model))
    if not unique_models:
        raise ValueError("At least one Graphiti LLM model name is required")

    clients = [
        (
            model,
            JsonTolerantOpenAIClient(
                config=LLMConfig(api_key=api_key, base_url=base_url, model=model)
            ),
        )
        for model in unique_models
    ]
    return FailoverLLMClient(clients, cooldown_seconds=cooldown_seconds)


def build_embedder(
    *,
    api_key: str,
    base_urls: Sequence[str],
    model: str,
    dimension: int,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
) -> FailoverEmbedder:
    """Build a failover embedder chain; every URL must serve the same model."""

    unique_urls = list(dict.fromkeys(url for url in base_urls if url))
    if not unique_urls:
        raise ValueError("At least one Graphiti embedder base URL is required")

    embedders = [
        (
            url,
            OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=api_key,
                    base_url=url,
                    embedding_model=model,
                    embedding_dim=dimension,
                )
            ),
        )
        for url in unique_urls
    ]
    return FailoverEmbedder(embedders, cooldown_seconds=cooldown_seconds)
