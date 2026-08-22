"""Tests for the LLM model failover chain and the graphiti embedder fallback.

Two independent chains share the same policy: transport/5xx/429 moves on and
parks the failed provider, 4xx propagates. Both are covered here.
"""

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.services.graphiti_backend.failover import (
    CooldownRegistry,
    FailoverEmbedder,
    FailoverLLMClient,
    is_failover_worthy,
)
from app.utils import llm_client as llm_client_module
from app.utils.llm_client import LLMClient, clear_model_cooldowns


def _request():
    return httpx.Request("POST", "https://proxy.invalid/v1/chat/completions")


def _status_error(status_code):
    response = httpx.Response(status_code, request=_request())
    return APIStatusError("boom", response=response, body=None)


@pytest.fixture(autouse=True)
def _reset_cooldowns():
    clear_model_cooldowns()
    yield
    clear_model_cooldowns()


# --------------------------------------------------------------------------
# Shared failure classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        (_status_error(500), True),
        (_status_error(503), True),
        (_status_error(429), True),
        (_status_error(408), True),
        (_status_error(400), False),
        (_status_error(401), False),
        (_status_error(404), False),
        (APIConnectionError(request=_request()), True),
        (APITimeoutError(request=_request()), True),
        (ConnectionError("socket closed"), True),
        (ValueError("bad payload"), False),
    ],
)
def test_failure_classification(error, expected):
    assert is_failover_worthy(error) is expected
    assert llm_client_module._is_failover_worthy(error) is expected


# --------------------------------------------------------------------------
# MiroFish LLMClient model chain
# --------------------------------------------------------------------------


def _build_llm_client(monkeypatch, models, behaviour):
    """Build an LLMClient whose chat completions follow ``behaviour``."""

    calls = []

    def fake_create_chat_completion(client, *, model, **kwargs):
        calls.append(model)
        outcome = behaviour(model)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm_client_module.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_client_module.Config, "LLM_BASE_URL", "https://proxy.invalid/v1")
    monkeypatch.setattr(llm_client_module.Config, "LLM_MODEL_NAME", models[0])
    monkeypatch.setattr(llm_client_module.Config, "LLM_MODEL_FALLBACKS", list(models[1:]))
    monkeypatch.setattr(
        llm_client_module, "create_chat_completion", fake_create_chat_completion
    )
    return LLMClient(), calls


def test_llm_uses_the_primary_model_when_it_works(monkeypatch):
    client, calls = _build_llm_client(
        monkeypatch, ["Kimi-k2.6", "claude-sonnet-4-6"], lambda model: "response"
    )

    result = client._create_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=100,
        response_format=None,
    )

    assert result == "response"
    assert calls == ["Kimi-k2.6"]


def test_llm_switches_to_the_next_model_after_a_server_error(monkeypatch):
    def behaviour(model):
        return _status_error(503) if model == "Kimi-k2.6" else "recovered"

    client, calls = _build_llm_client(
        monkeypatch, ["Kimi-k2.6", "claude-sonnet-4-6", "gpt-5.4-mini"], behaviour
    )

    result = client._create_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=100,
        response_format=None,
    )

    assert result == "recovered"
    assert calls == ["Kimi-k2.6", "claude-sonnet-4-6"]


def test_llm_does_not_switch_models_on_a_client_error(monkeypatch):
    client, calls = _build_llm_client(
        monkeypatch,
        ["Kimi-k2.6", "claude-sonnet-4-6"],
        lambda model: _status_error(400),
    )

    with pytest.raises(APIStatusError):
        client._create_completion(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
            response_format=None,
        )

    assert calls == ["Kimi-k2.6"]


def test_llm_skips_a_model_that_is_still_in_cooldown(monkeypatch):
    def behaviour(model):
        return _status_error(500) if model == "Kimi-k2.6" else "recovered"

    client, calls = _build_llm_client(
        monkeypatch, ["Kimi-k2.6", "claude-sonnet-4-6"], behaviour
    )
    request = dict(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=100,
        response_format=None,
    )

    client._create_completion(**request)
    client._create_completion(**request)

    # The second call must not retry the parked primary model.
    assert calls == ["Kimi-k2.6", "claude-sonnet-4-6", "claude-sonnet-4-6"]


def test_llm_raises_the_last_error_when_every_model_fails(monkeypatch):
    client, calls = _build_llm_client(
        monkeypatch,
        ["Kimi-k2.6", "claude-sonnet-4-6"],
        lambda model: _status_error(502),
    )

    with pytest.raises(APIStatusError):
        client._create_completion(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
            response_format=None,
        )

    assert calls == ["Kimi-k2.6", "claude-sonnet-4-6"]


def test_llm_reports_a_fully_parked_chain(monkeypatch):
    client, _ = _build_llm_client(
        monkeypatch, ["Kimi-k2.6"], lambda model: _status_error(500)
    )
    request = dict(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=100,
        response_format=None,
    )

    with pytest.raises(APIStatusError):
        client._create_completion(**request)
    with pytest.raises(RuntimeError, match="冷却"):
        client._create_completion(**request)


def test_llm_model_chain_deduplicates_the_primary_model(monkeypatch):
    client, _ = _build_llm_client(
        monkeypatch, ["Kimi-k2.6", "Kimi-k2.6", "gpt-5.4-mini"], lambda model: "ok"
    )

    assert client.model_chain == ["Kimi-k2.6", "gpt-5.4-mini"]


# --------------------------------------------------------------------------
# Cooldown registry
# --------------------------------------------------------------------------


def test_cooldown_registry_expires_entries():
    registry = CooldownRegistry(cooldown_seconds=0.0)
    registry.park("model-a")

    assert registry.is_parked("model-a") is False


def test_cooldown_registry_holds_and_clears():
    registry = CooldownRegistry(cooldown_seconds=60.0)
    registry.park("model-a")

    assert registry.is_parked("model-a") is True
    registry.clear()
    assert registry.is_parked("model-a") is False


# --------------------------------------------------------------------------
# Graphiti embedder fallback
# --------------------------------------------------------------------------


class _StubEmbedder:
    def __init__(self, error=None, vector=None):
        self._error = error
        self._vector = vector or [0.1, 0.2]
        self.calls = 0

    async def create(self, input_data):
        self.calls += 1
        if self._error:
            raise self._error
        return self._vector

    async def create_batch(self, input_data_list):
        self.calls += 1
        if self._error:
            raise self._error
        return [self._vector for _ in input_data_list]


@pytest.mark.asyncio
async def test_embedder_falls_back_to_the_local_endpoint():
    gpu = _StubEmbedder(error=APIConnectionError(request=_request()))
    local = _StubEmbedder(vector=[0.5, 0.6])
    embedder = FailoverEmbedder([("gpu", gpu), ("local", local)])

    assert await embedder.create("text") == [0.5, 0.6]
    assert (gpu.calls, local.calls) == (1, 1)
    # The parked GPU endpoint is skipped on the next call.
    assert await embedder.create_batch(["a", "b"]) == [[0.5, 0.6], [0.5, 0.6]]
    assert gpu.calls == 1


@pytest.mark.asyncio
async def test_embedder_propagates_a_client_error_without_failing_over():
    gpu = _StubEmbedder(error=_status_error(400))
    local = _StubEmbedder()
    embedder = FailoverEmbedder([("gpu", gpu), ("local", local)])

    with pytest.raises(APIStatusError):
        await embedder.create("text")

    assert local.calls == 0


@pytest.mark.asyncio
async def test_embedder_raises_when_every_endpoint_is_parked():
    gpu = _StubEmbedder(error=APIConnectionError(request=_request()))
    embedder = FailoverEmbedder([("gpu", gpu)])

    with pytest.raises(APIConnectionError):
        await embedder.create("text")
    with pytest.raises(RuntimeError, match="cooldown"):
        await embedder.create("text")


def test_embedder_requires_at_least_one_endpoint():
    with pytest.raises(ValueError, match="at least one embedder"):
        FailoverEmbedder([])


def test_llm_failover_client_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one client"):
        FailoverLLMClient([])
