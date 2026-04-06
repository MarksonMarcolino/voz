"""Kokoro TTS engine wrapper — fast, streaming-capable synthesis."""

import logging
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np

from src.config import KOKORO_LANG_CODES, KOKORO_SAMPLE_RATE, KOKORO_VOICES

# Required for Apple Silicon MPS support
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)


class KokoroEngine:
    """Fast TTS with streaming chunk generation.

    Kokoro outputs audio as a generator of numpy chunks (one per sentence/clause),
    which maps naturally to WebSocket streaming.
    """

    def __init__(self):
        self._pipelines: dict[str, object] = {}

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            logger.info(f"Loading Kokoro pipeline for lang_code={lang_code}...")
            from kokoro import KPipeline

            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id="hexgrad/Kokoro-82M"
            )
            logger.info(f"Kokoro pipeline loaded for {lang_code}")
        return self._pipelines[lang_code]

    def _resolve_voice(self, language: str, voice: str | None, gender: str = "female") -> tuple[str, str]:
        """Resolve language and voice to Kokoro lang_code and voice ID."""
        lang_code = KOKORO_LANG_CODES.get(language)
        if not lang_code:
            raise ValueError(
                f"Unsupported language: {language}. Available: {list(KOKORO_LANG_CODES.keys())}"
            )

        if voice:
            return lang_code, voice

        voices = KOKORO_VOICES.get(language, {})
        resolved = voices.get(gender)
        if not resolved:
            resolved = next(iter(voices.values())) if voices else None
        if not resolved:
            raise ValueError(f"No voice available for language={language}, gender={gender}")
        return lang_code, resolved

    def stream(
        self,
        text: str,
        language: str = "pt",
        voice: str | None = None,
        gender: str = "female",
    ) -> Generator[np.ndarray, None, None]:
        """Yield audio chunks as they're generated.

        Each chunk is a float32 numpy array at KOKORO_SAMPLE_RATE (24kHz).
        Chunks correspond to sentence/clause boundaries.
        """
        lang_code, voice_id = self._resolve_voice(language, voice, gender)
        pipeline = self._get_pipeline(lang_code)

        for _graphemes, _phonemes, audio in pipeline(text, voice=voice_id):
            yield audio

    def synthesize(
        self,
        text: str,
        language: str = "pt",
        voice: str | None = None,
        gender: str = "female",
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate speech and save to a WAV file."""
        import soundfile as sf

        chunks = list(self.stream(text, language, voice, gender))
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, KOKORO_SAMPLE_RATE)
        logger.info(f"Generated: {output_path}")
        return output_path
