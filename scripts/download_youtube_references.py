"""Download high-quality reference audio clips from YouTube.

Extracts clean, single-speaker segments for voice cloning.
For PoC/research use only (fair use).
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "reference_audio"
CACHE_DIR = BASE_DIR / ".yt_cache"

# YouTube sources: picked for clear single-speaker audio with strong accent
# Format: (accent_id, youtube_url, start_time_seconds, duration_seconds, description)
SOURCES = [
    # Mineiro: "51 frases que todo mineiro fala" — single speaker, clear accent
    ("br_mineiro", "https://www.youtube.com/watch?v=JX5luHglKcU", 15, 20,
     "51 frases que todo mineiro fala"),

    # Carioca: "Sotaque carioca: por que se fala chiado no Rio" — narrator with carioca accent
    ("br_carioca", "https://www.youtube.com/watch?v=kJUUNisO_hI", 30, 20,
     "Sotaque carioca explicado"),

    # Nordestino: "Aprenda a falar pernambuques em 8 minutos" — single speaker, strong accent
    ("br_nordestino", "https://www.youtube.com/watch?v=zmHvaG6eh1A", 20, 20,
     "Pernambuquês em 8 minutos"),

    # Gaúcho: "Sotaques do Brasil - Rio Grande do Sul" — single speaker explaining accent
    ("br_gaucho", "https://www.youtube.com/watch?v=490htEEOMmM", 15, 20,
     "Sotaque gaúcho explicado"),

    # Paulista: "Sotaques do Brasil - Minas Gerais" channel likely has SP too
    # Using a different source — clear paulista speaker
    ("br_paulista", "https://www.youtube.com/watch?v=IqFSljlB8xU", 10, 20,
     "Como se fala em São Paulo"),

    # Rioplatense: "Cómo hablar español de Argentina" — single speaker, clear porteño accent
    ("ar_rioplatense", "https://www.youtube.com/watch?v=slogyR0ixGA", 30, 20,
     "Cómo hablar español de Argentina"),
]


def download_clip(accent_id: str, url: str, start: int, duration: int, desc: str):
    """Download a YouTube video segment and extract clean audio."""
    dest = REFERENCE_DIR / f"{accent_id}.wav"
    if dest.exists():
        print(f"  {accent_id}: Already exists, skipping")
        return True

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    raw_audio = CACHE_DIR / f"{accent_id}_raw.wav"

    print(f"  {accent_id}: Downloading from YouTube...")
    print(f"    Source: {desc}")
    print(f"    Segment: {start}s - {start + duration}s")

    try:
        # Step 1: Download full audio
        subprocess.run([
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(raw_audio),
            "--no-playlist",
            "--quiet",
            url,
        ], check=True, capture_output=True, text=True)

        # Step 2: Extract segment with ffmpeg (mono, 24kHz for TTS)
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", str(raw_audio),
            "-ss", str(start),
            "-t", str(duration),
            "-ar", "24000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(dest),
        ], check=True, capture_output=True, text=True)

        # Verify output
        import soundfile as sf
        data, sr = sf.read(str(dest))
        actual_dur = len(data) / sr
        print(f"    Output: {actual_dur:.1f}s, {sr}Hz → {dest.name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e.stderr[:200] if e.stderr else str(e)}")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False
    finally:
        # Clean up raw download
        if raw_audio.exists():
            raw_audio.unlink()


def main():
    print("=" * 60)
    print("YouTube Reference Audio Downloader")
    print("=" * 60)
    print()
    print("Downloading 20s clips of single speakers per accent...")
    print("(For PoC/research use only)")
    print()

    results = {}
    for accent_id, url, start, duration, desc in SOURCES:
        ok = download_clip(accent_id, url, start, duration, desc)
        results[accent_id] = ok
        print()

    print("=" * 60)
    print("Results:")
    for accent_id, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {accent_id:20s} {status}")

    failed = [a for a, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} failed. You can manually adjust timestamps in this script.")
    else:
        print("\nAll clips downloaded! Run: uv run python -m scripts.validate_reference_audio")


if __name__ == "__main__":
    main()
