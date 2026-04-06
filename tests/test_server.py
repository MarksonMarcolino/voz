"""Tests for FastAPI server endpoints."""

import base64


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_accents_endpoint(client):
    response = client.get("/accents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 8


def test_synthesize_valid_request(client):
    response = client.post(
        "/synthesize",
        json={"text": "Bom dia", "accent": "br_female"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_synthesize_unknown_accent(client):
    response = client.post(
        "/synthesize",
        json={"text": "Bom dia", "accent": "xx_invalid"},
    )
    assert response.status_code == 400
    assert "Unknown accent" in response.json()["detail"]


def test_synthesize_empty_text(client):
    response = client.post(
        "/synthesize",
        json={"text": "", "accent": "br_female"},
    )
    assert response.status_code == 422


def test_synthesize_text_too_long(client):
    response = client.post(
        "/synthesize",
        json={"text": "x" * 2001, "accent": "br_female"},
    )
    assert response.status_code == 422


def test_synthesize_exaggeration_out_of_range(client):
    response = client.post(
        "/synthesize",
        json={"text": "Bom dia", "accent": "br_female", "exaggeration": 5.0},
    )
    assert response.status_code == 422


def test_synthesize_default_mode(client):
    response = client.post(
        "/synthesize",
        json={"text": "Bom dia", "accent": "br_female"},
    )
    assert response.status_code == 200


# --- WebSocket tests ---


def test_ws_synthesize_basic(ws_client):
    with ws_client.websocket_connect("/ws/synthesize") as ws:
        ws.send_json({"type": "synthesize", "text": "Bom dia", "language": "pt"})
        msg1 = ws.receive_json()
        assert msg1["type"] == "metadata"
        assert msg1["sample_rate"] == 24000
        assert msg1["encoding"] == "pcm_s16le"
        assert msg1["channels"] == 1

        msg2 = ws.receive_json()
        assert msg2["type"] == "audio"
        assert "data" in msg2
        assert msg2["chunk_index"] == 0

        msg3 = ws.receive_json()
        assert msg3["type"] == "done"
        assert msg3["total_chunks"] == 1


def test_ws_synthesize_unknown_language(ws_client):
    with ws_client.websocket_connect("/ws/synthesize") as ws:
        ws.send_json({"type": "synthesize", "text": "Hello", "language": "fr"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Unknown language" in msg["detail"]


def test_ws_synthesize_empty_text(ws_client):
    with ws_client.websocket_connect("/ws/synthesize") as ws:
        ws.send_json({"type": "synthesize", "text": "", "language": "pt"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Empty text" in msg["detail"]


def test_ws_synthesize_unknown_message_type(ws_client):
    with ws_client.websocket_connect("/ws/synthesize") as ws:
        ws.send_json({"type": "unknown"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Unknown message type" in msg["detail"]


def test_ws_synthesize_multiple_requests(ws_client):
    with ws_client.websocket_connect("/ws/synthesize") as ws:
        # First request
        ws.send_json({"type": "synthesize", "text": "Bom dia", "language": "pt"})
        assert ws.receive_json()["type"] == "metadata"
        assert ws.receive_json()["type"] == "audio"
        assert ws.receive_json()["type"] == "done"

        # Second request on same connection
        ws.send_json({"type": "synthesize", "text": "Hola", "language": "es"})
        assert ws.receive_json()["type"] == "metadata"
        assert ws.receive_json()["type"] == "audio"
        assert ws.receive_json()["type"] == "done"


def test_ws_audio_format(ws_client):
    with ws_client.websocket_connect("/ws/synthesize") as ws:
        ws.send_json({"type": "synthesize", "text": "Bom dia", "language": "pt"})
        ws.receive_json()  # metadata
        msg = ws.receive_json()  # audio

        pcm_bytes = base64.b64decode(msg["data"])
        # 16-bit mono PCM: byte count must be even
        assert len(pcm_bytes) % 2 == 0
        # 2400 float32 samples -> 2400 int16 samples -> 4800 bytes
        assert len(pcm_bytes) == 4800


def test_ws_voices_endpoint(ws_client):
    response = ws_client.get("/voices")
    assert response.status_code == 200
    voices = response.json()
    assert len(voices) == 4  # 2 languages x 2 genders
    assert all("language" in v and "voice" in v for v in voices)
