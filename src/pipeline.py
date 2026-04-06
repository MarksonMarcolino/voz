"""Combined TTS pipeline: Chatterbox → OpenVoice accent transfer."""

import logging
from pathlib import Path
from typing import Literal

from src.config import get_reference_path, ACCENTS
from src.tts_chatterbox import ChatterboxEngine

try:
    from src.tts_openvoice import OpenVoiceAccentConverter
    OPENVOICE_AVAILABLE = True
except ImportError:
    OPENVOICE_AVAILABLE = False

logger = logging.getLogger(__name__)

Mode = Literal["chatterbox_only", "pipeline"]


class TTSPipeline:
    """Combined TTS pipeline with accent control.

    Modes:
        - chatterbox_only: Chatterbox voice cloning (accent from reference clip)
        - pipeline: Chatterbox base speech → OpenVoice accent transfer
    """

    def __init__(self, device: str | None = None):
        self.chatterbox = ChatterboxEngine(device=device)
        self._openvoice = None
        self._device = device

    @property
    def openvoice(self):
        if self._openvoice is None:
            if not OPENVOICE_AVAILABLE:
                raise RuntimeError(
                    "OpenVoice V2 not installed. Install separately:\n"
                    "  pip install git+https://github.com/myshell-ai/OpenVoice.git"
                )
            self._openvoice = OpenVoiceAccentConverter(device=self._device)
        return self._openvoice

    def synthesize(
        self,
        text: str,
        accent_id: str,
        mode: Mode = "chatterbox_only",
        output_path: str | Path | None = None,
        exaggeration: float = 0.5,
    ) -> Path:
        """Generate accented speech.

        Args:
            text: Text to synthesize.
            accent_id: Target accent (e.g., "br_carioca", "ar_rioplatense").
            mode: "chatterbox_only" or "pipeline" (with OpenVoice accent transfer).
            output_path: Where to save output.
            exaggeration: Accent emphasis for Chatterbox (0.25-2.0).
        """
        ref_audio = get_reference_path(accent_id)
        language = ACCENTS[accent_id].language

        if mode == "chatterbox_only":
            return self.chatterbox.synthesize(
                text=text,
                reference_audio=ref_audio,
                language=language,
                output_path=output_path,
                exaggeration=exaggeration,
            )

        elif mode == "pipeline":
            # Step 1: Generate base speech with Chatterbox
            # Use low cfg_weight to reduce accent bleed from reference
            base_audio = self.chatterbox.synthesize(
                text=text,
                reference_audio=ref_audio,
                language=language,
                exaggeration=exaggeration,
                cfg_weight=0.3,
            )

            # Step 2: Apply accent via OpenVoice tone color converter
            return self.openvoice.convert_accent(
                audio_path=base_audio,
                reference_audio=ref_audio,
                output_path=output_path,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'chatterbox_only' or 'pipeline'.")

    @staticmethod
    def list_accents() -> list[dict]:
        return [
            {
                "id": a.id,
                "name": a.name,
                "language": a.language,
                "region": a.region,
            }
            for a in ACCENTS.values()
        ]
