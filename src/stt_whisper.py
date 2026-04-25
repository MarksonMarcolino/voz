"""Server-side speech-to-text using faster-whisper."""

import io
import logging

import numpy as np

from src.config import WHISPER_MODEL_SIZE, WHISPER_SAMPLE_RATE

logger = logging.getLogger(__name__)


class WhisperEngine:
    """Transcribe audio using faster-whisper (CTranslate2 backend)."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self._model = None
        self._model_size = model_size

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading faster-whisper model: {self._model_size}...")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device="auto",
                compute_type="default",
            )
            logger.info("Whisper model loaded")
        return self._model

    def transcribe(self, audio_bytes: bytes, language: str = "pt") -> str:
        """Transcribe PCM audio (16-bit signed, 16kHz, mono) to text.

        Args:
            audio_bytes: Raw PCM s16le bytes at 16kHz mono.
            language: Language code ("pt" or "es").

        Returns:
            Transcribed text string.
        """
        # Convert raw PCM bytes to float32 numpy array
        int16_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        float_audio = int16_audio.astype(np.float32) / 32768.0

        model = self._get_model()
        segments, info = model.transcribe(
            float_audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(segment.text.strip() for segment in segments)
        logger.info(f"STT: transcribed {len(float_audio)} samples ({len(float_audio)/WHISPER_SAMPLE_RATE:.1f}s) → \"{text[:80]}\"")
        return text
