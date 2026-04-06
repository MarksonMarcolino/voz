"""Async Ollama LLM client with streaming support."""

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, SYSTEM_PROMPTS

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when Ollama is unavailable or returns an error."""


async def stream_chat(
    prompt: str,
    language: str = "pt",
    system_prompt: str | None = None,
    model: str = OLLAMA_MODEL,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama chat API.

    Yields individual token strings as they arrive.
    Raises OllamaError if Ollama is unreachable.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["pt"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=5.0)) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as response:
                if response.status_code != 200:
                    raise OllamaError(f"Ollama returned status {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        yield content
                    if data.get("done"):
                        return
    except httpx.ConnectError:
        raise OllamaError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is it running? Start with: ollama serve"
        )
    except httpx.TimeoutException:
        raise OllamaError(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")


async def check_ollama_health() -> bool:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
