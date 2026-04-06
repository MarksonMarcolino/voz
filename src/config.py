"""Accent registry and model configuration."""

from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent.parent
REFERENCE_AUDIO_DIR = BASE_DIR / "reference_audio"
SAMPLES_DIR = BASE_DIR / "samples"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints_v2"

# Default device: MPS on Apple Silicon, CUDA if available, else CPU
def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Accent:
    id: str
    name: str
    language: str  # ISO 639-1
    region: str
    reference_file: str  # filename in reference_audio/


ACCENTS = {
    "br_female": Accent("br_female", "Brazilian Female", "pt", "Brazil", "br_female.wav"),
    "br_male": Accent("br_male", "Brazilian Male", "pt", "Brazil", "br_male.wav"),
    "ar_rioplatense": Accent("ar_rioplatense", "Rioplatense (Buenos Aires)", "es", "Argentina", "ar_rioplatense.wav"),
}

TEST_PHRASES = {
    "pt": [
        "Bom dia, tudo bem com você?",
        "Eu preciso verificar a conexão da fibra óptica.",
        "O técnico vai chegar em trinta minutos para fazer a instalação.",
    ],
    "es": [
        "Hola, ¿cómo andás?",
        "Necesito verificar la conexión de fibra óptica.",
        "El técnico va a llegar en treinta minutos para hacer la instalación.",
    ],
}


def get_reference_path(accent_id: str) -> Path:
    accent = ACCENTS.get(accent_id)
    if not accent:
        raise ValueError(f"Unknown accent: {accent_id}. Available: {list(ACCENTS.keys())}")
    path = REFERENCE_AUDIO_DIR / accent.reference_file
    if not path.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {path}\n"
            f"Please add a 10-30s WAV clip of a {accent.name} speaker."
        )
    return path
