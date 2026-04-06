"""Shared test fixtures. All ML models are mocked -- no GPU required."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import REFERENCE_AUDIO_DIR


@pytest.fixture
def tmp_wav(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file for testing."""
    import struct

    wav_path = tmp_path / "test.wav"
    # Minimal WAV: 44-byte header + 2 bytes of silence
    sample_rate = 24000
    num_samples = 2
    data_size = num_samples * 2  # 16-bit mono
    with open(wav_path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
    return wav_path


@pytest.fixture
def mock_chatterbox(tmp_wav: Path):
    """Mock ChatterboxEngine that returns a dummy WAV path."""
    engine = MagicMock()
    engine.synthesize.return_value = tmp_wav
    return engine


@pytest.fixture
def mock_pipeline(mock_chatterbox, tmp_wav: Path):
    """TTSPipeline with mocked engines."""
    with patch("src.pipeline.ChatterboxEngine", return_value=mock_chatterbox):
        from src.pipeline import TTSPipeline

        pipeline = TTSPipeline.__new__(TTSPipeline)
        pipeline.chatterbox = mock_chatterbox
        pipeline._openvoice = None
        pipeline._device = "cpu"
        return pipeline


@pytest.fixture
def client(mock_pipeline):
    """FastAPI TestClient with mocked pipeline."""
    import src.server as server_module

    server_module.pipeline = mock_pipeline
    from fastapi.testclient import TestClient

    with TestClient(server_module.app) as c:
        yield c
    server_module.pipeline = None
