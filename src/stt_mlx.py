"""Apple-MLX-backed Whisper STT engine — same interface as WhisperEngine."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.config import WHISPER_MLX_REPO, WHISPER_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Whisper processes audio in 30-second windows. mlx-whisper JIT-compiles a
# fresh kernel for each unique input shape; padding every clip to exactly
# 30 s lets MLX reuse the same compiled kernel on every call (~1.2 s steady
# state vs ~14-18 s if we let shapes vary).
_MLX_TARGET_LEN = 30 * WHISPER_SAMPLE_RATE


def _pad_to_30s(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= _MLX_TARGET_LEN:
        return audio[:_MLX_TARGET_LEN]
    out = np.zeros(_MLX_TARGET_LEN, dtype=np.float32)
    out[: len(audio)] = audio
    return out


class MlxWhisperEngine:
    """Whisper via Apple MLX. Drop-in for WhisperEngine.

    mlx-whisper is module-level (no model class to instantiate). The first
    transcribe() call downloads + JIT-compiles weights; we trigger that in
    _get_model() so the lifespan warm task can load it at startup.
    """

    def __init__(self, repo: str = WHISPER_MLX_REPO):
        self._repo = repo
        self._warmed = False
        # MLX uses a per-thread command queue; running transcribe across
        # different threads (asyncio's default ThreadPoolExecutor has many)
        # causes 5-15s of GPU contention overhead per call. A single
        # dedicated worker pins MLX to one thread for consistent ~1s timing.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-whisper")

    def _get_model(self):
        """Pre-load + JIT-compile the model. Called by the startup warm task."""
        if self._warmed:
            return
        logger.info(f"Loading MLX whisper model: {self._repo}...")
        # Warm with a full 30-second silent buffer — this is the same shape
        # we'll use on every real call, so MLX caches the compiled kernel.
        silent = np.zeros(_MLX_TARGET_LEN, dtype=np.float32)
        self._transcribe_array(silent, language="pt", initial_prompt=None)
        self._warmed = True
        logger.info("MLX whisper model loaded")

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "pt",
        initial_prompt: str | None = None,
    ) -> str:
        t0 = time.perf_counter()
        int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        float_audio = int16.astype(np.float32) / 32768.0
        # Pad to 30s so the JIT-compiled kernel is reused (huge speedup).
        # Whisper trims trailing silence internally, so padding is safe.
        padded = _pad_to_30s(float_audio)
        t1 = time.perf_counter()
        text = self._transcribe_array(padded, language, initial_prompt)
        t2 = time.perf_counter()
        logger.info(f"STT(mlx) timing: prep={t1-t0:.3f}s transcribe={t2-t1:.3f}s")
        prompt_preview = (
            (initial_prompt[:60] + "…")
            if initial_prompt and len(initial_prompt) > 60
            else (initial_prompt or "")
        )
        logger.info(
            f"STT(mlx): {len(float_audio)} samples "
            f"({len(float_audio)/WHISPER_SAMPLE_RATE:.1f}s) "
            f'prompt="{prompt_preview}" → "{text[:80]}"'
        )
        return text

    def _transcribe_array(
        self,
        float_audio: np.ndarray,
        language: str,
        initial_prompt: str | None,
    ) -> str:
        import mlx_whisper

        # mlx-whisper uses greedy decoding only (no beam search). That's fine
        # for short conversational utterances and is part of why it's fast.
        result = mlx_whisper.transcribe(
            float_audio,
            path_or_hf_repo=self._repo,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            temperature=0.0,
            verbose=None,
        )
        return (result.get("text") or "").strip()
