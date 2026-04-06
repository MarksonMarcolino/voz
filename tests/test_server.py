"""Tests for FastAPI server endpoints."""


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
