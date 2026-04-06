"""Benchmark TTS latency on the current device.

Measures:
  - Model load time (cold start)
  - Time-to-first-audio (TTFA) per utterance
  - Total generation time per utterance
  - Audio duration vs generation time (real-time factor)

Usage:
  uv run python scripts/benchmark_latency.py
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ACCENTS, TEST_PHRASES, get_device, get_reference_path


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def get_audio_duration(wav_path: Path) -> float:
    """Get duration in seconds of a WAV file."""
    import torchaudio as ta
    info = ta.info(str(wav_path))
    return info.num_frames / info.sample_rate


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Accents: {len(ACCENTS)}")
    print()

    # --- Cold start: model load ---
    print("Loading model (cold start)...")
    t0 = time.perf_counter()

    from src.tts_chatterbox import ChatterboxEngine
    engine = ChatterboxEngine(device=device)
    _ = engine.model  # trigger lazy load

    load_time = time.perf_counter() - t0
    print(f"Model load: {format_ms(load_time)}")
    print()

    # --- Benchmark matrix ---
    # Test with 2 accents (pt + es) and 3 phrase lengths
    test_cases = [
        ("short_pt", "br_carioca", "pt", "Bom dia, tudo bem?"),
        ("medium_pt", "br_carioca", "pt", "Eu preciso verificar a conexão da fibra óptica."),
        ("long_pt", "br_carioca", "pt",
         "O técnico vai chegar em trinta minutos para fazer a instalação. "
         "Por favor, verifique se o roteador está ligado e conectado."),
        ("short_es", "ar_rioplatense", "es", "Hola, ¿cómo andás?"),
        ("medium_es", "ar_rioplatense", "es", "Necesito verificar la conexión de fibra óptica."),
        ("long_es", "ar_rioplatense", "es",
         "El técnico va a llegar en treinta minutos para hacer la instalación. "
         "Por favor, verifique que el router esté encendido y conectado."),
    ]

    # Warm-up run (first inference is always slower)
    print("Warm-up run...")
    ref = get_reference_path("br_carioca")
    engine.synthesize(text="Teste.", reference_audio=ref, language="pt")
    print()

    # Benchmark
    print(f"{'Case':<12} {'Chars':>5} {'Gen Time':>10} {'Audio Dur':>10} {'RTF':>6} {'Speed':>8}")
    print("-" * 62)

    results = []
    for name, accent_id, lang, text in test_cases:
        ref = get_reference_path(accent_id)

        t_start = time.perf_counter()
        wav_path = engine.synthesize(
            text=text,
            reference_audio=ref,
            language=lang,
            exaggeration=0.5,
        )
        gen_time = time.perf_counter() - t_start

        audio_dur = get_audio_duration(wav_path)
        rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")
        speed = f"{1/rtf:.1f}x" if rtf > 0 else "N/A"

        print(f"{name:<12} {len(text):>5} {format_ms(gen_time):>10} {audio_dur:>9.1f}s {rtf:>6.2f} {speed:>8}")

        results.append({
            "name": name,
            "chars": len(text),
            "gen_time_s": gen_time,
            "audio_dur_s": audio_dur,
            "rtf": rtf,
        })

        # Clean up temp file
        wav_path.unlink(missing_ok=True)

    # Summary
    print()
    avg_gen = sum(r["gen_time_s"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_dur = sum(r["audio_dur_s"] for r in results) / len(results)

    print(f"Average generation time: {format_ms(avg_gen)}")
    print(f"Average audio duration:  {avg_dur:.1f}s")
    print(f"Average RTF:             {avg_rtf:.2f}")
    print(f"Average speed:           {1/avg_rtf:.1f}x real-time")
    print()

    # Verdict
    if avg_rtf < 0.5:
        print("VERDICT: Excellent. Fast enough for real-time streaming.")
    elif avg_rtf < 1.0:
        print("VERDICT: Good. Faster than real-time, usable for conversational AI with buffering.")
    elif avg_rtf < 2.0:
        print("VERDICT: Marginal. Slower than real-time. Users will notice delay.")
        print("  Consider: smaller model, GPU upgrade, or switching to a faster TTS engine.")
    else:
        print("VERDICT: Too slow for conversational use.")
        print("  Consider: CUDA GPU, distilled model, or a streaming-native TTS (e.g., Kokoro, MARS5).")


if __name__ == "__main__":
    main()
