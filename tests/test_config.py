"""Tests for accent registry and configuration."""

import pytest

from src.config import ACCENTS, TEST_PHRASES, REFERENCE_AUDIO_DIR, get_reference_path


def test_accent_count():
    assert len(ACCENTS) == 8


def test_all_accents_have_reference_files():
    for accent_id, accent in ACCENTS.items():
        path = REFERENCE_AUDIO_DIR / accent.reference_file
        assert path.exists(), f"Missing reference audio for {accent_id}: {path}"


def test_all_accent_languages_have_test_phrases():
    languages = {a.language for a in ACCENTS.values()}
    for lang in languages:
        assert lang in TEST_PHRASES, f"No test phrases for language: {lang}"
        assert len(TEST_PHRASES[lang]) > 0


def test_get_reference_path_valid():
    path = get_reference_path("br_female")
    assert path.name == "br_female.wav"
    assert path.exists()


def test_get_reference_path_unknown_accent():
    with pytest.raises(ValueError, match="Unknown accent"):
        get_reference_path("xx_invalid")


def test_accent_dataclass_fields():
    accent = ACCENTS["br_female"]
    assert accent.id == "br_female"
    assert accent.name == "Brazilian Female"
    assert accent.language == "pt"
    assert accent.region == "Brazil"
    assert accent.reference_file == "br_female.wav"


def test_get_device_returns_valid_string():
    from src.config import get_device

    device = get_device()
    assert device in ("cuda", "mps", "cpu")
