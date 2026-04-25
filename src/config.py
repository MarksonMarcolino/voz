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
    "br_carioca": Accent("br_carioca", "Brazilian Carioca (Rio)", "pt", "Brazil", "br_carioca.wav"),
    "br_gaucho": Accent("br_gaucho", "Brazilian Gaúcho (RS)", "pt", "Brazil", "br_gaucho.wav"),
    "br_mineiro": Accent("br_mineiro", "Brazilian Mineiro (MG)", "pt", "Brazil", "br_mineiro.wav"),
    "br_nordestino": Accent("br_nordestino", "Brazilian Nordestino (NE)", "pt", "Brazil", "br_nordestino.wav"),
    "br_paulista": Accent("br_paulista", "Brazilian Paulista (SP)", "pt", "Brazil", "br_paulista.wav"),
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


# Kokoro TTS engine config (fast, streaming-capable)
KOKORO_VOICES = {
    "pt": {"female": "pf_dora", "male": "pm_alex"},
    "es": {"female": "ef_dora", "male": "em_alex"},
}
KOKORO_LANG_CODES = {"pt": "p", "es": "e"}
KOKORO_SAMPLE_RATE = 24000


# Ollama LLM config
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_TIMEOUT = 60.0

SYSTEM_PROMPTS = {
    "pt": (
        "Voce e Sofia, assistente de atendimento ao cliente de telecomunicacoes. "
        "Fale portugues brasileiro. Respostas concisas (2-3 frases)."
    ),
    "es": (
        "Eres Sofia, asistente de atencion al cliente de telecomunicaciones. "
        "Habla espanol. Respuestas concisas (2-3 oraciones)."
    ),
}

SENTENCE_MIN_LENGTH = 10
MAX_HISTORY_TURNS = 10

# Whisper STT config
WHISPER_MODEL_SIZE = "base"
WHISPER_SAMPLE_RATE = 16000


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
