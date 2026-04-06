"""Download reference audio for accents not in CORAA.

Sources:
- ar_rioplatense: OpenSLR #61 (Google Argentine Spanish, CC BY-SA 4.0)
- br_carioca: Mozilla Common Voice (CC-0) — filter by accent
- br_gaucho: Mozilla Common Voice (CC-0) — filter by accent

For PoC/research use only.
"""

import io
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "reference_audio"
CACHE_DIR = BASE_DIR / ".cache_missing"

# OpenSLR #61: Argentine Spanish female speakers (~1.2GB)
# Using female because it's larger/more diverse
OPENSLR_URL = "https://openslr.trmal.net/resources/61/es_ar_female.zip"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download with progress."""
    print(f"Downloading {desc or url}...")
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    print()


def extract_rioplatense():
    """Extract a 15s+ clip from OpenSLR Argentine Spanish dataset."""
    dest = REFERENCE_DIR / "ar_rioplatense.wav"
    if dest.exists():
        print(f"ar_rioplatense: Already exists ({dest})")
        return

    zip_path = CACHE_DIR / "es_ar_female.zip"
    download_file(OPENSLR_URL, zip_path, "OpenSLR Argentine Spanish female (~1.2GB)")

    print("Extracting Argentine Spanish clips...")
    target_duration = 15.0
    all_audio = []
    sample_rate = None
    total_duration = 0.0

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        wav_files = sorted([n for n in zf.namelist() if n.endswith(".wav")])
        print(f"  Found {len(wav_files)} WAV files")

        for wav_file in wav_files:
            if total_duration >= target_duration:
                break
            try:
                with zf.open(wav_file) as src:
                    data, sr = sf.read(io.BytesIO(src.read()))
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    continue
                all_audio.append(data)
                total_duration += len(data) / sr
            except Exception:
                continue

    if all_audio and sample_rate:
        concatenated = np.concatenate(all_audio)
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), concatenated, sample_rate)
        print(f"  ar_rioplatense: {total_duration:.1f}s → {dest.name}")
    else:
        print("  ar_rioplatense: FAILED")


def extract_common_voice_accents():
    """Try to extract carioca and gaúcho clips from Common Voice via HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n  ERROR: 'datasets' not installed. Run: uv add datasets")
        return

    # Map CV accent labels we want to try
    target_accents = {
        "br_carioca": ["Rio de Janeiro", "Carioca", "carioca", "rio de janeiro", "RJ"],
        "br_gaucho": ["Rio Grande do Sul", "Gaúcho", "gaúcho", "gaucho", "RS", "Sul"],
    }

    print("\nLoading Common Voice Portuguese metadata (streaming)...")
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "pt",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed to load Common Voice: {e}")
        print("  You may need to accept the dataset license at:")
        print("  https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        return

    # First pass: discover available accent labels
    print("  Scanning for accent labels (first 5000 samples)...")
    accent_counts = {}
    samples_by_accent = {aid: [] for aid in target_accents}
    checked = 0

    for sample in ds:
        checked += 1
        if checked > 5000:
            break

        accent = (sample.get("accent") or "").strip()
        if accent:
            accent_counts[accent] = accent_counts.get(accent, 0) + 1

            # Check if this matches any target
            for accent_id, keywords in target_accents.items():
                for kw in keywords:
                    if kw.lower() in accent.lower():
                        samples_by_accent[accent_id].append(sample)
                        break

    print(f"  Scanned {checked} samples. Accent labels found:")
    for label, count in sorted(accent_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")

    # Extract and concatenate matching clips
    for accent_id, samples in samples_by_accent.items():
        dest = REFERENCE_DIR / f"{accent_id}.wav"
        if dest.exists():
            print(f"\n  {accent_id}: Already exists")
            continue

        if not samples:
            print(f"\n  {accent_id}: No matching clips found in Common Voice")
            continue

        print(f"\n  {accent_id}: Found {len(samples)} matching clips, concatenating...")
        all_audio = []
        sample_rate = None
        total_duration = 0.0

        for sample in samples:
            if total_duration >= 15.0:
                break
            try:
                audio = sample["audio"]
                data = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                if sample_rate is None:
                    sample_rate = sr
                all_audio.append(data)
                total_duration += len(data) / sr
            except Exception:
                continue

        if all_audio and sample_rate:
            concatenated = np.concatenate(all_audio)
            sf.write(str(dest), concatenated, sample_rate)
            print(f"  {accent_id}: {total_duration:.1f}s → {dest.name}")
        else:
            print(f"  {accent_id}: Failed to extract audio")


def main():
    print("=" * 60)
    print("Download Missing Reference Audio")
    print("=" * 60)
    print()

    # Step 1: Argentine Spanish from OpenSLR
    print("--- Argentine Spanish (OpenSLR #61) ---")
    extract_rioplatense()

    # Step 2: Try Common Voice for carioca and gaúcho
    print("\n--- Brazilian Accents (Mozilla Common Voice) ---")
    extract_common_voice_accents()

    # Summary
    print("\n" + "=" * 60)
    print("Reference Audio Status:")
    for name in ["br_paulista", "br_carioca", "br_nordestino", "br_gaucho", "br_mineiro", "ar_rioplatense"]:
        path = REFERENCE_DIR / f"{name}.wav"
        if path.exists():
            data, sr = sf.read(str(path))
            dur = len(data) / sr
            print(f"  {name:20s} OK ({dur:.1f}s, {sr}Hz)")
        else:
            print(f"  {name:20s} MISSING")


if __name__ == "__main__":
    main()
