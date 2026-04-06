"""Tests for TTSPipeline with mocked engines."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import ACCENTS


def test_synthesize_chatterbox_only(mock_pipeline, tmp_wav):
    result = mock_pipeline.synthesize(
        text="Bom dia",
        accent_id="br_female",
        mode="chatterbox_only",
        exaggeration=0.5,
    )
    assert result == tmp_wav
    mock_pipeline.chatterbox.synthesize.assert_called_once()
    call_kwargs = mock_pipeline.chatterbox.synthesize.call_args
    assert call_kwargs.kwargs.get("language") == "pt" or call_kwargs[1].get("language") == "pt"


def test_synthesize_pipeline_mode(mock_pipeline, tmp_wav):
    mock_openvoice = MagicMock()
    mock_openvoice.convert_accent.return_value = tmp_wav
    mock_pipeline._openvoice = mock_openvoice

    with patch("src.pipeline.OPENVOICE_AVAILABLE", True):
        result = mock_pipeline.synthesize(
            text="Bom dia",
            accent_id="br_female",
            mode="pipeline",
            exaggeration=0.5,
        )

    assert result == tmp_wav
    mock_pipeline.chatterbox.synthesize.assert_called_once()
    # Pipeline mode uses low cfg_weight to reduce accent bleed
    call_kwargs = mock_pipeline.chatterbox.synthesize.call_args
    assert call_kwargs.kwargs.get("cfg_weight") == 0.3 or call_kwargs[1].get("cfg_weight") == 0.3
    mock_openvoice.convert_accent.assert_called_once()


def test_synthesize_invalid_mode(mock_pipeline):
    with pytest.raises(ValueError, match="Unknown mode"):
        mock_pipeline.synthesize(
            text="Bom dia",
            accent_id="br_female",
            mode="invalid",
        )


def test_openvoice_not_available_raises(mock_pipeline):
    mock_pipeline._openvoice = None
    with patch("src.pipeline.OPENVOICE_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="OpenVoice V2 not installed"):
            mock_pipeline.synthesize(
                text="Bom dia",
                accent_id="br_female",
                mode="pipeline",
            )


def test_list_accents_structure():
    from src.pipeline import TTSPipeline

    accents = TTSPipeline.list_accents()
    assert isinstance(accents, list)
    assert len(accents) == len(ACCENTS)
    for accent in accents:
        assert set(accent.keys()) == {"id", "name", "language", "region"}


def test_list_accents_contains_all_ids():
    from src.pipeline import TTSPipeline

    accent_ids = {a["id"] for a in TTSPipeline.list_accents()}
    assert accent_ids == set(ACCENTS.keys())
