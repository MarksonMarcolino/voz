"""Server-side speech-to-text using faster-whisper."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from src.config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_MODEL_SIZE,
    WHISPER_SAMPLE_RATE,
    WHISPER_STATIC_PROMPT,
)

if TYPE_CHECKING:
    from src.history import ConversationHistory

logger = logging.getLogger(__name__)

_MD_STRIP = re.compile(r"[*_`#>]+")


def _strip_for_prompt(text: str) -> str:
    """Lightweight markdown removal for prompts (Whisper sees plain text)."""
    return _MD_STRIP.sub("", text).strip()


def build_stt_prompt(
    history: ConversationHistory | None,
    language: str,
    max_chars: int = 200,
) -> str | None:
    """Build a Whisper initial_prompt from static glossary + last assistant turn.

    The prompt biases the decoder toward domain vocabulary and the bot's
    last reply (which usually contains the proper nouns the user is about
    to repeat). Whisper caps prompts at ~224 tokens; we cap on chars.
    """
    parts: list[str] = []
    glossary = WHISPER_STATIC_PROMPT.get(language)
    if glossary:
        parts.append(glossary)

    if history is not None:
        # ConversationHistory stores messages in _messages; reach in here
        # rather than adding another helper method just for this.
        for msg in reversed(history._messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                parts.append(_strip_for_prompt(msg["content"]))
                break

    if not parts:
        return None

    prompt = " ".join(parts)
    if len(prompt) > max_chars:
        # Keep the tail — that's the most recent context the user is replying to
        prompt = prompt[-max_chars:]
    return prompt


class WhisperEngine:
    """Transcribe audio using faster-whisper (CTranslate2 backend)."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self._model = None
        self._model_size = model_size
        self._compute_type = compute_type

    def _get_model(self):
        if self._model is None:
            logger.info(
                f"Loading faster-whisper model: {self._model_size} "
                f"(compute_type={self._compute_type})..."
            )
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device="auto",
                compute_type=self._compute_type,
            )
            logger.info("Whisper model loaded")
        return self._model

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "pt",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe PCM audio (16-bit signed, 16kHz, mono) to text.

        Args:
            audio_bytes: Raw PCM s16le bytes at 16kHz mono.
            language: Language code ("pt" or "es").
            initial_prompt: Optional text used to bias decoding toward
                specific vocabulary (proper nouns, domain words, prior turn).

        Returns:
            Transcribed text string.
        """
        int16_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        float_audio = int16_audio.astype(np.float32) / 32768.0

        model = self._get_model()
        # beam_size=1 (greedy) is fast and fine for short conversational utterances;
        # bumped via config if quality regresses.
        segments, _info = model.transcribe(
            float_audio,
            language=language,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,
            initial_prompt=initial_prompt,
            # For short utterances: don't condition on prior decoded text
            # (prevents hallucinated continuations) and stay greedy.
            condition_on_previous_text=False,
            temperature=0.0,
        )

        text = " ".join(segment.text.strip() for segment in segments)
        prompt_preview = (initial_prompt[:60] + "…") if initial_prompt and len(initial_prompt) > 60 else (initial_prompt or "")
        logger.info(
            f"STT: {len(float_audio)} samples ({len(float_audio)/WHISPER_SAMPLE_RATE:.1f}s) "
            f"prompt=\"{prompt_preview}\" → \"{text[:80]}\""
        )
        return text
