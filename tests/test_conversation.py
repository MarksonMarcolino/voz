"""Tests for SentenceBuffer and conversational WebSocket endpoint."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.conversation import SentenceBuffer


# --- SentenceBuffer unit tests ---


def test_buffer_single_sentence():
    buf = SentenceBuffer(min_length=5)
    results = []
    for token in ["Hello", " world", ".", " Next"]:
        results.extend(buf.add(token))
    assert results == ["Hello world."]
    assert buf.flush() == "Next"


def test_buffer_multiple_sentences():
    buf = SentenceBuffer(min_length=5)
    results = buf.add("First sentence. Second sentence. Third")
    assert results == ["First sentence.", "Second sentence."]
    assert buf.flush() == "Third"


def test_buffer_question_mark():
    buf = SentenceBuffer(min_length=5)
    results = buf.add("Como vai? Tudo bem.")
    assert results == ["Como vai?"]
    assert buf.flush() == "Tudo bem."


def test_buffer_exclamation():
    buf = SentenceBuffer(min_length=5)
    results = buf.add("Bom dia! Como estas?")
    assert results == ["Bom dia!"]
    assert buf.flush() == "Como estas?"


def test_buffer_min_length_prevents_tiny_splits():
    buf = SentenceBuffer(min_length=10)
    results = buf.add("Hi. How are you doing today?")
    # "Hi." is 3 chars, below min_length
    assert results == []
    assert buf.flush() == "Hi. How are you doing today?"


def test_buffer_flush_empty():
    buf = SentenceBuffer(min_length=5)
    assert buf.flush() is None


def test_buffer_unicode_content():
    buf = SentenceBuffer(min_length=5)
    results = buf.add("A conexão está estável. Posso verificar mais.")
    assert len(results) == 1
    assert results[0] == "A conexão está estável."
    assert buf.flush() == "Posso verificar mais."


def test_buffer_streaming_tokens():
    """Simulate token-by-token streaming from LLM."""
    buf = SentenceBuffer(min_length=5)
    tokens = ["A ", "conexão ", "está ", "ok.", " Posso ", "ajudar."]
    results = []
    for t in tokens:
        results.extend(buf.add(t))
    remaining = buf.flush()
    assert results == ["A conexão está ok."]
    assert remaining == "Posso ajudar."


# --- WebSocket conversation integration tests ---


@pytest.fixture
def mock_stream_chat():
    """Mock stream_chat to yield predefined tokens."""

    async def fake_stream(*args, **kwargs):
        for token in ["A conexão ", "está estável.", " Posso ", "verificar."]:
            yield token

    return fake_stream


@pytest.fixture
def conversation_client(mock_pipeline, mock_kokoro_engine, mock_stream_chat):
    """FastAPI TestClient with mocked Ollama and Kokoro."""
    import src.server as server_module

    server_module.pipeline = mock_pipeline
    server_module.kokoro_engine = mock_kokoro_engine
    from fastapi.testclient import TestClient

    with patch("src.conversation.stream_chat", side_effect=mock_stream_chat):
        with TestClient(server_module.app) as c:
            yield c
    server_module.pipeline = None
    server_module.kokoro_engine = None


def test_ws_conversation_basic(conversation_client):
    with conversation_client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "conversation", "text": "Ola", "language": "pt"})

        msg = ws.receive_json()
        assert msg["type"] == "metadata"
        assert msg["sample_rate"] == 24000

        # Should get transcript + audio for each sentence
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] == "done":
                break

        types = [m["type"] for m in messages]
        assert "transcript" in types
        assert "audio" in types
        assert types[-1] == "done"


def test_ws_conversation_unknown_type(conversation_client):
    with conversation_client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "unknown"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Unknown type" in msg["detail"]


def test_ws_conversation_empty_text(conversation_client):
    with conversation_client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "conversation", "text": "", "language": "pt"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Empty text" in msg["detail"]


def test_ws_conversation_unknown_language(conversation_client):
    with conversation_client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "conversation", "text": "Ola", "language": "fr"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Unknown language" in msg["detail"]


@pytest.fixture
def conversation_client_ollama_down(mock_pipeline, mock_kokoro_engine):
    """Client where Ollama is simulated as unreachable."""
    import src.server as server_module
    from src.llm_ollama import OllamaError

    server_module.pipeline = mock_pipeline
    server_module.kokoro_engine = mock_kokoro_engine

    async def failing_stream(*args, **kwargs):
        raise OllamaError("Cannot connect to Ollama at http://localhost:11434.")
        # Make it a generator
        yield  # noqa: unreachable

    from fastapi.testclient import TestClient

    with patch("src.conversation.stream_chat", side_effect=failing_stream):
        with TestClient(server_module.app) as c:
            yield c
    server_module.pipeline = None
    server_module.kokoro_engine = None


def test_ws_conversation_ollama_down(conversation_client_ollama_down):
    with conversation_client_ollama_down.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "conversation", "text": "Ola", "language": "pt"})

        msg = ws.receive_json()
        assert msg["type"] == "metadata"

        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Cannot connect" in msg["detail"]
