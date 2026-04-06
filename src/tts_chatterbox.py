"""Chatterbox TTS engine wrapper."""

import logging
import tempfile
from pathlib import Path

import torchaudio as ta

from src.config import get_device

logger = logging.getLogger(__name__)


class ChatterboxEngine:
    """High-quality multilingual TTS with voice cloning."""

    def __init__(self, device: str | None = None):
        self.device = device or get_device()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading Chatterbox Multilingual model on {self.device}...")
            # Patch: PerthImplicitWatermarker fails to build on Apple Silicon.
            # Replace with DummyWatermarker so model loads without watermarking.
            import perth
            if perth.PerthImplicitWatermarker is None:
                logger.warning("PerthImplicitWatermarker unavailable, using DummyWatermarker")
                perth.PerthImplicitWatermarker = perth.DummyWatermarker

            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            # Patch torch.load to map CUDA checkpoints to current device
            import torch
            _original_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs.setdefault("map_location", self.device)
                return _original_load(*args, **kwargs)
            torch.load = _patched_load
            try:
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            finally:
                torch.load = _original_load
            logger.info("Chatterbox Multilingual model loaded")
        return self._model

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        language: str = "pt",
        output_path: str | Path | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Path:
        """Generate speech cloning the voice from reference audio.

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference audio clip (10-30s WAV).
            language: ISO 639-1 language code (pt, es, en, etc.).
            output_path: Where to save output. Auto-generated if None.
            exaggeration: Accent/emotion emphasis (0.25-2.0, default 0.5).
            cfg_weight: Generation strength (0.0-1.0). Use 0.0 for cross-language
                        transfer to reduce accent bleed from reference clip.
        """
        wav = self.model.generate(
            text,
            language_id=language,
            audio_prompt_path=str(reference_audio),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), wav, self.model.sr)
        logger.info(f"Generated: {output_path}")
        return output_path
