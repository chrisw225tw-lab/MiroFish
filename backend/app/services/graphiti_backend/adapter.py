"""Drop-in replacement for the Zep Cloud client, backed by local Graphiti.

``GraphitiAdapter`` exposes the same ``.graph`` / ``.batch`` namespaces the Zep
SDK does, so ``utils.zep.get_zep_client()`` can hand either object to the rest
of the backend.
"""

from __future__ import annotations

import os
import threading

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.config import LLMConfig

from ...config import Config
from ...utils.logger import get_logger
from .engine import GraphitiEngine
from .failover import build_embedder, build_llm_client
from .namespaces import BatchNamespace, GraphNamespace

logger = get_logger("mirofish.graphiti.adapter")

GRAPH_STATE_DIRNAME = "graph_state"


class GraphitiAdapter:
    """Zep-shaped facade over a :class:`GraphitiEngine`."""

    def __init__(self, engine: GraphitiEngine) -> None:
        self._engine = engine
        self.graph = GraphNamespace(engine)
        self.batch = BatchNamespace(engine)

    @property
    def engine(self) -> GraphitiEngine:
        return self._engine

    def close(self) -> None:
        self._engine.close()


def _graph_state_dir() -> str:
    state_dir = os.path.join(Config.UPLOAD_FOLDER, GRAPH_STATE_DIRNAME)
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def build_graphiti_adapter() -> GraphitiAdapter:
    """Build an adapter from ``Config``. Raises if the config is incomplete."""

    api_key = Config.GRAPHITI_LLM_API_KEY or Config.LLM_API_KEY
    base_url = Config.GRAPHITI_LLM_BASE_URL or Config.LLM_BASE_URL
    primary_model = Config.GRAPHITI_LLM_MODEL_NAME or Config.LLM_MODEL_NAME
    fallback_models = (
        Config.GRAPHITI_LLM_MODEL_FALLBACKS
        if Config.GRAPHITI_LLM_MODEL_NAME
        else Config.LLM_MODEL_FALLBACKS
    )
    if not api_key:
        raise ValueError("GRAPHITI_LLM_API_KEY 或 LLM_API_KEY 未配置")
    if not Config.GRAPHITI_EMBEDDER_BASE_URL:
        raise ValueError("GRAPHITI_EMBEDDER_BASE_URL 未配置")

    llm_client = build_llm_client(
        api_key=api_key,
        base_url=base_url,
        models=[primary_model, *fallback_models],
    )
    embedder = build_embedder(
        api_key=Config.GRAPHITI_EMBEDDER_API_KEY or "ollama",
        base_urls=[
            Config.GRAPHITI_EMBEDDER_BASE_URL,
            Config.GRAPHITI_EMBEDDER_FALLBACK_BASE_URL,
        ],
        model=Config.GRAPHITI_EMBEDDER_MODEL,
        dimension=Config.GRAPHITI_EMBEDDER_DIM,
    )
    # Searches use RRF/MMR recipes, so the cross-encoder is never exercised; it
    # is still supplied explicitly so graphiti does not build a default client
    # that reads OPENAI_API_KEY from the environment.
    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(api_key=api_key, base_url=base_url, model=primary_model)
    )

    engine = GraphitiEngine(
        falkordb_uri=Config.GRAPHITI_FALKORDB_URI,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        state_dir=_graph_state_dir(),
        database=Config.GRAPHITI_FALKORDB_DATABASE,
    )
    logger.info(
        "Graphiti adapter built (models=%s, embedder=%s)",
        llm_client.model_chain,
        embedder.endpoint_chain,
    )
    return GraphitiAdapter(engine)


_adapter: GraphitiAdapter | None = None
_adapter_lock = threading.Lock()


def get_graphiti_adapter() -> GraphitiAdapter:
    """Return the process-wide adapter, building it on first use."""

    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = build_graphiti_adapter()
        return _adapter


def clear_graphiti_adapter_cache() -> None:
    """Drop the cached adapter. Intended for tests and reconfiguration."""

    global _adapter
    with _adapter_lock:
        adapter, _adapter = _adapter, None
    if adapter is not None:
        adapter.close()
