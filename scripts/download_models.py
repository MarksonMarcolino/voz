"""Download model checkpoints for Chatterbox and OpenVoice V2."""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints_v2"


def download_openvoice_v2():
    """Download OpenVoice V2 checkpoints from HuggingFace."""
    print("Downloading OpenVoice V2 checkpoints...")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # OpenVoice V2 checkpoints are hosted on HuggingFace
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="myshell-ai/OpenVoiceV2",
            local_dir=str(CHECKPOINTS_DIR),
        )
        print(f"OpenVoice V2 checkpoints saved to {CHECKPOINTS_DIR}")
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="myshell-ai/OpenVoiceV2",
            local_dir=str(CHECKPOINTS_DIR),
        )
        print(f"OpenVoice V2 checkpoints saved to {CHECKPOINTS_DIR}")


def verify_chatterbox():
    """Verify Chatterbox model can be loaded (downloads on first use)."""
    print("Verifying Chatterbox model availability...")
    try:
        from chatterbox.tts import ChatterboxTTS
        print("Chatterbox package available (model downloads on first synthesis)")
    except ImportError:
        print("ERROR: chatterbox-tts not installed. Run: uv sync")
        sys.exit(1)


def main():
    print("=" * 50)
    print("TTS PoC — Model Download")
    print("=" * 50)
    print()

    verify_chatterbox()
    print()
    download_openvoice_v2()

    print()
    print("All models ready!")


if __name__ == "__main__":
    main()
