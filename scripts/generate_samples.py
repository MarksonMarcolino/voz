"""Generate accent samples for quality comparison."""

import logging
from pathlib import Path

from src.config import ACCENTS, TEST_PHRASES, SAMPLES_DIR
from src.pipeline import TTSPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_all_samples():
    pipeline = TTSPipeline()

    modes = ["chatterbox_only", "pipeline"]

    for accent_id, accent in ACCENTS.items():
        phrases = TEST_PHRASES.get(accent.language, [])
        if not phrases:
            logger.warning(f"No test phrases for language {accent.language}, skipping {accent_id}")
            continue

        for mode in modes:
            for i, phrase in enumerate(phrases):
                output_dir = SAMPLES_DIR / mode
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{accent_id}_phrase{i + 1}.wav"

                try:
                    logger.info(f"Generating: {accent_id} / {mode} / phrase {i + 1}")
                    pipeline.synthesize(
                        text=phrase,
                        accent_id=accent_id,
                        mode=mode,
                        output_path=output_file,
                    )
                    logger.info(f"  → {output_file}")
                except FileNotFoundError as e:
                    logger.warning(f"  Skipped (no reference audio): {e}")
                    break
                except Exception as e:
                    logger.error(f"  Failed: {e}")

    print()
    print("Sample generation complete!")
    print(f"Output directory: {SAMPLES_DIR}")
    print()
    print("Next steps:")
    print("  1. Listen to samples in samples/ directory")
    print("  2. Compare with ElevenLabs baseline (add to samples/elevenlabs/)")
    print("  3. Run: uv run python scripts/compare_quality.py")


if __name__ == "__main__":
    generate_all_samples()
