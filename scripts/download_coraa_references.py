"""Download CORAA dataset and extract reference audio clips per accent.

CORAA provides Brazilian Portuguese speech with accent labels:
- São Paulo (cap.) → paulista
- Minas Gerais → mineiro
- Recife → nordestino
- São Paulo (int.) → paulista interior variant

Downloads the dev set (~1.3GB) and picks the best clip per accent
(longest clean audio with high quality votes).
"""

import csv
import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

METADATA_URL = "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/metadata_dev_final.csv"
AUDIO_URL = "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/dev.zip"

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "reference_audio"

# Map CORAA accent labels to our accent IDs
ACCENT_MAP = {
    "São Paulo (cap.)": "br_paulista",
    "Minas Gerais": "br_mineiro",
    "Recife": "br_nordestino",
}


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress."""
    print(f"Downloading {desc or url}...")
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
    print()


def find_best_clips(metadata_path: Path) -> dict[str, dict]:
    """Find the best audio clip per accent from metadata.

    Criteria: high up_votes, zero down_votes, no noise/second voice problems,
    prefer longer audio paths (larger file names often = longer clips).
    """
    candidates: dict[str, list] = {accent_id: [] for accent_id in ACCENT_MAP.values()}

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            coraa_accent = row["accent"]
            if coraa_accent not in ACCENT_MAP:
                continue

            accent_id = ACCENT_MAP[coraa_accent]
            up = int(row["up_votes"])
            down = int(row["down_votes"])
            noise = int(row["votes_for_noise_or_low_voice"])
            second_voice = int(row["votes_for_second_voice"])

            # Quality filter: high agreement, no issues
            if down > 0 or noise > 0 or second_voice > 0:
                continue
            if up < 2:
                continue

            candidates[accent_id].append({
                "file_path": row["file_path"],
                "text": row["text"],
                "up_votes": up,
                "accent": coraa_accent,
            })

    # Sort by quality (votes) then text length
    for accent_id in candidates:
        candidates[accent_id].sort(key=lambda c: (c["up_votes"], len(c["text"])), reverse=True)

    return candidates


def extract_and_concatenate(zip_path: Path, candidates: dict[str, list], target_duration: float = 15.0):
    """Extract and concatenate clips per accent to reach target duration."""
    import numpy as np
    import soundfile as sf

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting and concatenating clips from {zip_path}...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for accent_id, clips in candidates.items():
            if not clips:
                print(f"  WARNING: No clean clips for {accent_id}")
                continue

            dest_path = REFERENCE_DIR / f"{accent_id}.wav"
            all_audio = []
            sample_rate = None
            total_duration = 0.0
            used = 0

            for clip in clips:
                if total_duration >= target_duration:
                    break
                try:
                    with zf.open(clip["file_path"]) as src:
                        audio_data = io.BytesIO(src.read())
                    data, sr = sf.read(audio_data)
                    if sample_rate is None:
                        sample_rate = sr
                    elif sr != sample_rate:
                        continue  # Skip clips with different sample rate
                    all_audio.append(data)
                    total_duration += len(data) / sr
                    used += 1
                except Exception as e:
                    continue

            if all_audio and sample_rate:
                concatenated = np.concatenate(all_audio)
                sf.write(str(dest_path), concatenated, sample_rate)
                print(f"  {accent_id}: {total_duration:.1f}s from {used} clips → {dest_path.name}")
            else:
                print(f"  {accent_id}: FAILED (no valid audio)")


def main():
    print("=" * 60)
    print("CORAA Reference Audio Downloader")
    print("=" * 60)
    print()
    print("Available accents in CORAA dev set:")
    for coraa, our_id in ACCENT_MAP.items():
        print(f"  {coraa:25s} → {our_id}")
    print()
    print("NOT in CORAA (need separate sources):")
    print("  br_carioca (Rio de Janeiro)")
    print("  br_gaucho (South)")
    print("  ar_rioplatense (Argentina)")
    print()

    # Step 1: Download metadata
    metadata_path = BASE_DIR / ".coraa_cache" / "metadata_dev_final.csv"
    download_file(METADATA_URL, metadata_path, "CORAA dev metadata")

    # Step 2: Find best clips per accent
    print("Finding best clips per accent...")
    candidates = find_best_clips(metadata_path)
    for accent_id, clips in candidates.items():
        print(f"  {accent_id}: {len(clips)} clean candidates")
    print()

    # Step 3: Download audio zip
    zip_path = BASE_DIR / ".coraa_cache" / "dev.zip"
    download_file(AUDIO_URL, zip_path, "CORAA dev audio (~1.3GB)")

    # Step 4: Extract and concatenate clips to reach 15s+ per accent
    extract_and_concatenate(zip_path, candidates, target_duration=15.0)

    print()
    print("Done! Reference clips saved to reference_audio/")
    print()
    print("Still needed (not in CORAA):")
    missing = {"br_carioca", "br_gaucho", "ar_rioplatense"}
    for m in sorted(missing):
        ref = REFERENCE_DIR / f"{m}.wav"
        status = "EXISTS" if ref.exists() else "MISSING"
        print(f"  {m}: {status}")
    print()
    print("For missing accents, source 15-20s clips from YouTube:")
    print("  yt-dlp -x --audio-format wav 'URL' -o raw.wav")
    print("  ffmpeg -i raw.wav -ss 00:01:30 -t 15 -ar 24000 -ac 1 reference_audio/br_carioca.wav")


if __name__ == "__main__":
    main()
