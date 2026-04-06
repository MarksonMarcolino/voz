"""Validate reference audio clips for TTS voice cloning."""

import sys
from pathlib import Path

from src.config import ACCENTS, REFERENCE_AUDIO_DIR


def validate():
    print("Reference Audio Validation")
    print("=" * 40)
    print()

    missing = []
    issues = []

    for accent_id, accent in ACCENTS.items():
        path = REFERENCE_AUDIO_DIR / accent.reference_file
        status = "OK"

        if not path.exists():
            status = "MISSING"
            missing.append(accent)
        else:
            try:
                import soundfile as sf
                data, sr = sf.read(str(path))
                duration = len(data) / sr

                if duration < 10:
                    status = f"TOO SHORT ({duration:.1f}s, need 10-30s)"
                    issues.append((accent, status))
                elif duration > 60:
                    status = f"TOO LONG ({duration:.1f}s, recommend 10-30s)"
                    issues.append((accent, status))
                elif sr < 16000:
                    status = f"LOW SAMPLE RATE ({sr}Hz, recommend 24kHz+)"
                    issues.append((accent, status))
                else:
                    status = f"OK ({duration:.1f}s, {sr}Hz)"
            except Exception as e:
                status = f"ERROR: {e}"
                issues.append((accent, status))

        print(f"  {accent_id:20s} {status}")

    print()
    if missing:
        print(f"{len(missing)} missing clips:")
        for a in missing:
            print(f"  - {REFERENCE_AUDIO_DIR / a.reference_file}")
            print(f"    Need: 10-30s WAV clip of a {a.name} speaker")
        print()
        print("Tips for sourcing reference audio:")
        print("  - YouTube interviews with native speakers from each region")
        print("  - Record yourself or ask native speakers")
        print("  - Clean, single speaker, minimal background noise")
        print("  - WAV format, 24kHz+ sample rate")

    if not missing and not issues:
        print("All reference clips validated!")

    return len(missing) == 0 and len(issues) == 0


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
