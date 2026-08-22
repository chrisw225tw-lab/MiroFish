"""A graphiti LLM client that tolerates proxies which ignore response_format.

``OpenAIGenericClient`` assumes the endpoint honours ``response_format`` and
returns bare JSON. Subscription-backed proxies frequently drop that parameter,
so the model answers with a Markdown-fenced object and graphiti's
``json.loads`` fails. This client states the schema in the prompt and parses
the reply the same forgiving way ``utils.llm_client.chat_json`` does.
"""

from __future__ import annotations

import json
import re
import types
import typing
from typing import Any

import openai
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from ...utils.llm_client import clean_chat_text
from ...utils.logger import get_logger

logger = get_logger("mirofish.graphiti.llm")

DEFAULT_MODEL = "gpt-4o-mini"


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a model reply that may be fenced or wrapped in prose."""

    cleaned = clean_chat_text(raw or "")
    if not cleaned:
        raise ValueError("LLM returned an empty response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the first complete JSON object embedded in the text.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"LLM response is not JSON: {cleaned[:200]}")


def _is_string_field(annotation: Any) -> bool:
    """Return whether a model field is declared as a (possibly optional) string."""

    if annotation is str:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return args == [str]
    return False


def normalize_to_schema(
    payload: dict[str, Any], response_model: type[BaseModel] | None
) -> dict[str, Any]:
    """Repair the two ways a model commonly mis-shapes a schema-driven reply.

    1. Echoing the JSON Schema envelope (``{"properties": {...}}``) instead of
       an instance of it.
    2. Nesting an object where the schema declares a plain string — FalkorDB
       rejects non-primitive property values outright, so those are flattened
       to JSON text rather than lost.
    """

    if response_model is None:
        return payload

    fields = response_model.model_fields
    nested = payload.get("properties")
    if (
        "properties" not in fields
        and isinstance(nested, dict)
        and any(key in fields for key in nested)
    ):
        logger.debug("Unwrapping schema-shaped LLM reply for %s", response_model.__name__)
        payload = nested

    # graphiti stores the raw reply, not a validated dump, so a key the model
    # invented outside the ontology would be persisted as-is. Pydantic ignores
    # extras on validation; drop them here so storage agrees with the schema.
    allows_extra = response_model.model_config.get("extra") == "allow"
    repaired = {
        key: value
        for key, value in payload.items()
        if allows_extra or key in fields
    }
    dropped = set(payload) - set(repaired)
    if dropped:
        logger.debug(
            "Dropping undeclared attribute(s) %s from %s",
            sorted(dropped),
            response_model.__name__,
        )

    for name, field in fields.items():
        value = repaired.get(name)
        if isinstance(value, (dict, list)) and _is_string_field(field.annotation):
            repaired[name] = json.dumps(value, ensure_ascii=False)
    return repaired


class JsonTolerantOpenAIClient(OpenAIGenericClient):
    """``OpenAIGenericClient`` that also works without server-side JSON mode."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: Any = None,
        max_tokens: int = 16384,
    ) -> None:
        super().__init__(config=config, cache=cache, client=client, max_tokens=max_tokens)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        openai_messages: list[dict[str, str]] = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role in ("user", "system"):
                openai_messages.append({"role": message.role, "content": message.content})

        if response_model is not None and openai_messages:
            # The endpoint may drop response_format, so the schema also has to
            # travel in the prompt for the reply to have the expected shape.
            schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            openai_messages[-1]["content"] += (
                "\n\nRespond with a single JSON object matching this schema, "
                f"with no surrounding prose or Markdown fence:\n\n{schema}"
            )

        response_format: dict[str, Any] = {"type": "json_object"}
        if response_model is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(response_model, "__name__", "structured_response"),
                    "schema": response_model.model_json_schema(),
                },
            }

        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )
        except openai.RateLimitError as error:
            raise RateLimitError from error
        except openai.BadRequestError as error:
            if not _mentions_response_format(error):
                raise
            logger.info("Endpoint rejected response_format; retrying without it")
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        payload = parse_json_response(response.choices[0].message.content or "")
        return normalize_to_schema(payload, response_model)


def _mentions_response_format(error: Exception) -> bool:
    return "response_format" in str(error).lower()
