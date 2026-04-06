"""Benchmark Kokoro and Piper TTS latency.

Measures time-to-audio, generation time, and real-time factor
for both engines with BR-PT and ES text.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/benchmark_alternatives.py
"""

import time
import sys
import os
import tempfile
import wave
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

TEST_CASES = [
    ("short_pt", "p", "Bom dia, tudo bem?"),
    ("medium_pt", "p", "Eu preciso verificar a conexão da fibra óptica."),
    ("long_pt", "p",
     "O técnico vai chegar em trinta minutos para fazer a instalação. "
     "Por favor, verifique se o roteador está ligado e conectado."),
    ("short_es", "e", "Hola, ¿cómo andás?"),
    ("medium_es", "e", "Necesito verificar la conexión de fibra óptica."),
    ("long_es", "e",
     "El técnico va a llegar en treinta minutos para hacer la instalación. "
     "Por favor, verifique que el router esté encendido y conectado."),
]


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def get_wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def benchmark_kokoro():
    print("=" * 70)
    print("KOKORO TTS BENCHMARK")
    print("=" * 70)

    print("Loading Kokoro pipeline...")
    t0 = time.perf_counter()
    from kokoro import KPipeline
    import soundfile as sf

    # Pre-load both language pipelines
    pipeline_pt = KPipeline(lang_code="p")
    pipeline_es = KPipeline(lang_code="e")
    load_time = time.perf_counter() - t0
    print(f"Model load: {format_ms(load_time)}")
    print()

    # Warm-up
    print("Warm-up run...")
    for _, _, audio in pipeline_pt("Teste.", voice="pf_dora"):
        pass
    print()

    voices = {"p": "pf_dora", "e": "ef_dora"}
    pipelines = {"p": pipeline_pt, "e": pipeline_es}

    print(f"{'Case':<12} {'Chars':>5} {'Gen Time':>10} {'Audio Dur':>10} {'RTF':>6} {'Speed':>8}")
    print("-" * 62)

    results = []
    for name, lang, text in TEST_CASES:
        pipeline = pipelines[lang]
        voice = voices[lang]
        tmp = tempfile.mktemp(suffix=".wav")

        t_start = time.perf_counter()
        audio_chunks = []
        for _, _, audio in pipeline(text, voice=voice):
            audio_chunks.append(audio)
        gen_time = time.perf_counter() - t_start

        # Save to get duration
        import numpy as np
        full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        sf.write(tmp, full_audio, 24000)
        audio_dur = get_wav_duration(tmp)
        os.unlink(tmp)

        rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")
        speed = f"{1/rtf:.1f}x" if rtf > 0 else "N/A"

        print(f"{name:<12} {len(text):>5} {format_ms(gen_time):>10} {audio_dur:>9.1f}s {rtf:>6.2f} {speed:>8}")
        results.append({"name": name, "gen_time_s": gen_time, "audio_dur_s": audio_dur, "rtf": rtf})

    avg_gen = sum(r["gen_time_s"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_dur = sum(r["audio_dur_s"] for r in results) / len(results)

    print()
    print(f"Average generation time: {format_ms(avg_gen)}")
    print(f"Average audio duration:  {avg_dur:.1f}s")
    print(f"Average RTF:             {avg_rtf:.2f}")
    print(f"Average speed:           {1/avg_rtf:.1f}x real-time")
    return avg_rtf


def benchmark_piper():
    print()
    print("=" * 70)
    print("PIPER TTS BENCHMARK")
    print("=" * 70)

    print("Loading Piper voices...")
    t0 = time.perf_counter()

    from piper import PiperVoice

    voices_dir = Path(__file__).parent.parent / "piper-voices"
    voice_pt = PiperVoice.load(
        str(voices_dir / "pt_BR-faber-medium.onnx"),
        config_path=str(voices_dir / "pt_BR-faber-medium.onnx.json"),
    )
    voice_es = PiperVoice.load(
        str(voices_dir / "es_ES-davefx-medium.onnx"),
        config_path=str(voices_dir / "es_ES-davefx-medium.onnx.json"),
    )

    load_time = time.perf_counter() - t0
    print(f"Model load: {format_ms(load_time)}")
    print()

    def piper_synthesize_to_file(voice, text, path):
        with wave.open(path, "wb") as wf:
            voice.synthesize_wav(text, wf)

    # Warm-up
    print("Warm-up run...")
    tmp = tempfile.mktemp(suffix=".wav")
    piper_synthesize_to_file(voice_pt, "Teste.", tmp)
    os.unlink(tmp)
    print()

    voices = {"p": voice_pt, "e": voice_es}

    piper_test_cases = [
        ("short_pt", "p", "Bom dia, tudo bem?"),
        ("medium_pt", "p", "Eu preciso verificar a conexão da fibra óptica."),
        ("long_pt", "p",
         "O técnico vai chegar em trinta minutos para fazer a instalação. "
         "Por favor, verifique se o roteador está ligado e conectado."),
        ("short_es", "e", "Hola, ¿cómo andás?"),
        ("medium_es", "e", "Necesito verificar la conexión de fibra óptica."),
        ("long_es", "e",
         "El técnico va a llegar en treinta minutos para hacer la instalación. "
         "Por favor, verifique que el router esté encendido y conectado."),
    ]

    print(f"{'Case':<12} {'Chars':>5} {'Gen Time':>10} {'Audio Dur':>10} {'RTF':>6} {'Speed':>8}")
    print("-" * 62)

    results = []
    for name, lang, text in piper_test_cases:
        voice = voices[lang]
        tmp = tempfile.mktemp(suffix=".wav")

        t_start = time.perf_counter()
        piper_synthesize_to_file(voice, text, tmp)
        gen_time = time.perf_counter() - t_start

        audio_dur = get_wav_duration(tmp)
        os.unlink(tmp)

        rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")
        speed = f"{1/rtf:.1f}x" if rtf > 0 else "N/A"

        print(f"{name:<12} {len(text):>5} {format_ms(gen_time):>10} {audio_dur:>9.1f}s {rtf:>6.2f} {speed:>8}")
        results.append({"name": name, "gen_time_s": gen_time, "audio_dur_s": audio_dur, "rtf": rtf})

    avg_gen = sum(r["gen_time_s"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_dur = sum(r["audio_dur_s"] for r in results) / len(results)

    print()
    print(f"Average generation time: {format_ms(avg_gen)}")
    print(f"Average audio duration:  {avg_dur:.1f}s")
    print(f"Average RTF:             {avg_rtf:.2f}")
    print(f"Average speed:           {1/avg_rtf:.1f}x real-time")
    return avg_rtf


def main():
    import platform
    import subprocess

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                            capture_output=True, text=True).stdout.strip()) // (1024**3)
    print(f"Hardware: {chip}, {ram}GB RAM")
    print(f"Python:   {platform.python_version()}")
    print()

    kokoro_rtf = benchmark_kokoro()
    piper_rtf = benchmark_piper()

    # Comparison
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Engine':<15} {'Avg RTF':>10} {'Avg Speed':>12} {'Verdict'}")
    print("-" * 55)

    for name, rtf in [("Chatterbox", 47.95), ("Kokoro", kokoro_rtf), ("Piper", piper_rtf)]:
        speed = f"{1/rtf:.1f}x" if rtf > 0 else "N/A"
        if rtf < 0.3:
            verdict = "Excellent for real-time"
        elif rtf < 1.0:
            verdict = "Good for conversational"
        elif rtf < 2.0:
            verdict = "Marginal"
        else:
            verdict = "Too slow"
        print(f"{name:<15} {rtf:>10.2f} {speed:>12} {verdict}")

    print()
    print("RTF = Real-Time Factor. <1.0 means faster than real-time.")
    print("For conversational AI, target RTF < 0.5 (2x+ real-time).")


if __name__ == "__main__":
    main()
