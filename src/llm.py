"""Gemini client for TravelJoy AI.

Provides text and structured (Pydantic) generation with model fallback.
Every request has a timeout, so a slow or unresponsive model falls through
to the next configured model instead of blocking the caller.

Usage:
    from src.llm import generate_text, generate_structured
"""
import logging
import os
from functools import lru_cache
from typing import TypeVar

import httpx2
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_MS = 30_000
TRANSIENT_ERROR_CODES = (429, 500, 503, 504)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Return a shared Gemini client, created on first use."""
    # Lazy creation keeps imports (e.g. in tests) from requiring an API key.
    return genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def _configured_models() -> list[str]:
    """Return the model names from .env, in fallback order."""
    models = [m for m in (os.getenv("GEMINI_MODEL"), os.getenv("GEMINI_FALLBACK_MODEL")) if m]
    if not models:
        raise RuntimeError("No models configured. Set GEMINI_MODEL in .env.")
    return models


def _generate(prompt: str, config: types.GenerateContentConfig) -> types.GenerateContentResponse:
    """Send the prompt to each configured model until one responds.

    Transient errors and timeouts move on to the next model; any other
    API error is raised immediately.
    """
    for model in _configured_models():
        try:
            response = _get_client().models.generate_content(
                model=model, contents=prompt, config=config
            )
            logger.info("Answered by %s", model)
            return response
        except errors.APIError as e:
            if e.code not in TRANSIENT_ERROR_CODES:
                raise
            logger.warning("%s unavailable (%s), trying next model", model, e.code)
        except httpx2.TimeoutException:
            # google-genai uses httpx2; its timeouts are not APIError and not httpx.TimeoutException.
            logger.warning("%s timed out after %d s, trying next model", model, REQUEST_TIMEOUT_MS // 1000)
    raise RuntimeError("All configured models are unavailable. Try again later.")


def generate_text(prompt: str, system_instruction: str | None = None) -> str:
    """Return the model's plain-text answer to the prompt."""
    config = types.GenerateContentConfig(system_instruction=system_instruction)
    response = _generate(prompt, config)
    if not response.text:
        raise RuntimeError("Model returned an empty response.")
    return response.text


def generate_structured(prompt: str, schema: type[SchemaT]) -> SchemaT:
    """Return the model's answer validated against the given Pydantic schema."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = _generate(prompt, config)
    if not isinstance(response.parsed, schema):
        raise RuntimeError("Model response did not match the expected schema.")
    return response.parsed