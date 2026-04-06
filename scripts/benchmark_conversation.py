"""Measure end-to-end conversation latency.

Connects to ws /ws/conversation, sends a text prompt,
and measures time-to-first-audio and total generation time.

Usage:
  uv run python scripts/benchmark_conversation.py
"""

import asyncio
import json
import time

import websockets


async def benchmark(text: str, language: str = "pt", runs: int = 3):
    uri = "ws://localhost:8000/ws/conversation"

    print(f"Text: \"{text}\"")
    print(f"Language: {language}")
    print(f"Runs: {runs}")
    print()

    results = []

    for run in range(1, runs + 1):
        async with websockets.connect(uri) as ws:
            t_send = time.perf_counter()

            await ws.send(json.dumps({
                "type": "conversation",
                "text": text,
                "language": language,
            }))

            t_first_transcript = None
            t_first_audio = None
            t_done = None
            total_chunks = 0
            sentences = []

            while True:
                msg = json.loads(await ws.recv())
                now = time.perf_counter()

                if msg["type"] == "metadata":
                    continue
                elif msg["type"] == "transcript":
                    if t_first_transcript is None:
                        t_first_transcript = now
                    sentences.append(msg["text"])
                elif msg["type"] == "audio":
                    if t_first_audio is None:
                        t_first_audio = now
                    total_chunks += 1
                elif msg["type"] == "done":
                    t_done = now
                    break
                elif msg["type"] == "error":
                    print(f"  ERROR: {msg['detail']}")
                    break

            if t_first_audio and t_done:
                ttft = (t_first_transcript - t_send) if t_first_transcript else None
                ttfa = t_first_audio - t_send
                total = t_done - t_send

                results.append({"ttft": ttft, "ttfa": ttfa, "total": total, "chunks": total_chunks})

                print(f"  Run {run}:")
                if ttft:
                    print(f"    Time to first transcript: {ttft*1000:.0f}ms")
                print(f"    Time to first audio:      {ttfa*1000:.0f}ms")
                print(f"    Total time:               {total*1000:.0f}ms")
                print(f"    Audio chunks:             {total_chunks}")
                print(f"    Sentences:                {len(sentences)}")
                for i, s in enumerate(sentences):
                    print(f"      [{i}] {s}")
                print()

    if results:
        avg_ttfa = sum(r["ttfa"] for r in results) / len(results)
        avg_total = sum(r["total"] for r in results) / len(results)
        print("=" * 50)
        print(f"Average time to first audio: {avg_ttfa*1000:.0f}ms")
        print(f"Average total time:          {avg_total*1000:.0f}ms")


async def main():
    test_cases = [
        ("pt", "Como está a conexão de fibra?"),
        ("es", "¿Cómo está la conexión de fibra?"),
    ]

    for lang, text in test_cases:
        print("=" * 50)
        await benchmark(text, lang, runs=2)
        print()


if __name__ == "__main__":
    asyncio.run(main())
