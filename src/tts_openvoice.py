"""OpenVoice V2 tone color converter wrapper."""

import logging
import tempfile
from pathlib import Path

from src.config import CHECKPOINTS_DIR, get_device

logger = logging.getLogger(__name__)


class OpenVoiceAccentConverter:
    """Applies accent/tone characteristics from reference audio to generated speech."""

    def __init__(self, device: str | None = None, checkpoint_dir: Path | None = None):
        self.device = device or get_device()
        self.checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            logger.info(f"Loading OpenVoice V2 converter on {self.device}...")
            from openvoice.api import ToneColorConverter

            config_path = self.checkpoint_dir / "config.json"
            ckpt_path = self.checkpoint_dir / "checkpoint.pth"

            if not config_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"OpenVoice V2 checkpoints not found in {self.checkpoint_dir}.\n"
                    "Run: uv run python scripts/download_models.py"
                )

            self._converter = ToneColorConverter(str(config_path), device=self.device)
            self._converter.load_ckpt(str(ckpt_path))
            logger.info("OpenVoice V2 converter loaded")
        return self._converter

    def convert_accent(
        self,
        audio_path: str | Path,
        reference_audio: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Apply tone color from reference audio to input speech.

        Args:
            audio_path: Input speech (e.g., from Chatterbox).
            reference_audio: Reference clip with target accent (10-30s WAV).
            output_path: Where to save output. Auto-generated if None.
        """
        src_se = self.converter.extract_se(str(audio_path))
        tgt_se = self.converter.extract_se(str(reference_audio))

        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.converter.convert(
            audio_src_path=str(audio_path),
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=str(output_path),
        )

        logger.info(f"Accent converted: {output_path}")
        return output_path
