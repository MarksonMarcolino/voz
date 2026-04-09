"""Pipelined conversation: LLM streaming → sentence buffer → TTS streaming → WebSocket."""

import asyncio
import base64
import logging
import re
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket

from src.config import KOKORO_SAMPLE_RATE, SENTENCE_MIN_LENGTH
from src.llm_ollama import OllamaError, stream_chat
from src.tts_kokoro import KokoroEngine

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SentenceBuffer:
    """Accumulates streaming tokens and yields complete sentences."""

    def __init__(self, min_length: int = SENTENCE_MIN_LENGTH):
        self._buffer = ""
        self._min_length = min_length

    def add(self, token: str) -> list[str]:
        """Add a token, return any complete sentences."""
        self._buffer += token
        sentences = []

        while True:
            match = _SENTENCE_BOUNDARY.search(self._buffer)
            if not match:
                break
            end = match.start() + 1  # include the punctuation
            candidate = self._buffer[:end].strip()
            remainder = self._buffer[match.end():]

            if len(candidate) >= self._min_length:
                sentences.append(candidate)
                self._buffer = remainder
            else:
                break

        return sentences

    def flush(self) -> str | None:
        """Flush remaining buffer content as final sentence."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None


@dataclass
class _SentenceReady:
    text: str
    index: int


@dataclass
class _AudioChunk:
    audio: np.ndarray
    sentence_index: int


def _audio_to_pcm_base64(audio) -> str:
    """Convert float32 audio (numpy or torch tensor) to base64-encoded 16-bit PCM."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


_SENTINEL = None


async def run_conversation(
    websocket: WebSocket,
    text: str,
    language: str,
    engine: KokoroEngine,
    voice: str | None = None,
    gender: str = "female",
) -> None:
    """Run the full LLM → TTS → WebSocket pipeline with streaming overlap."""

    sentence_queue: asyncio.Queue[_SentenceReady | None] = asyncio.Queue()
    audio_queue: asyncio.Queue[_AudioChunk | None] = asyncio.Queue()
    error_event = asyncio.Event()
    t_start = time.perf_counter()

    # --- Task 1: LLM streaming + sentence splitting ---
    async def llm_streamer():
        try:
            buffer = SentenceBuffer()
            sentence_index = 0

            async for token in stream_chat(text, language):
                if error_event.is_set():
                    return
                sentences = buffer.add(token)
                for s in sentences:
                    logger.info(f"LLM: sentence {sentence_index} ready: \"{s[:60]}\"")
                    await sentence_queue.put(_SentenceReady(text=s, index=sentence_index))
                    sentence_index += 1

            remaining = buffer.flush()
            if remaining:
                logger.info(f"LLM: final sentence {sentence_index}: \"{remaining[:60]}\"")
                await sentence_queue.put(_SentenceReady(text=remaining, index=sentence_index))

        except OllamaError as e:
            logger.error(f"Ollama error: {e}")
            await websocket.send_json({"type": "error", "detail": str(e)})
            error_event.set()
        except Exception as e:
            logger.error(f"LLM streamer error: {e}", exc_info=True)
            error_event.set()
        finally:
            await sentence_queue.put(_SENTINEL)

    # --- Task 2: TTS worker (Kokoro in thread pool) ---
    async def tts_worker():
        loop = asyncio.get_event_loop()
        try:
            while True:
                item = await sentence_queue.get()
                if item is None or error_event.is_set():
                    break

                sentence = item
                logger.info(f"TTS [{sentence.index}]: synthesizing \"{sentence.text[:60]}\"")

                # Send transcript before audio
                await websocket.send_json({
                    "type": "transcript",
                    "text": sentence.text,
                    "sentence_index": sentence.index,
                })

                # Run Kokoro in thread pool with timeout
                text_to_speak = sentence.text
                sent_lang = language
                sent_voice = voice
                sent_gender = gender

                def _synthesize():
                    return list(engine.stream(text_to_speak, sent_lang, sent_voice, sent_gender))

                try:
                    chunks = await asyncio.wait_for(
                        loop.run_in_executor(None, _synthesize),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"TTS [{sentence.index}]: timed out after 30s")
                    await websocket.send_json({
                        "type": "error",
                        "detail": f"TTS timed out for sentence: \"{sentence.text[:40]}...\"",
                    })
                    continue

                if not chunks:
                    logger.warning(f"TTS [{sentence.index}]: no audio generated")
                    continue

                for chunk in chunks:
                    logger.info(f"TTS [{sentence.index}]: audio chunk, {len(chunk) if hasattr(chunk, '__len__') else '?'} samples")
                    await audio_queue.put(_AudioChunk(
                        audio=chunk,
                        sentence_index=sentence.index,
                    ))

        except Exception as e:
            logger.error(f"TTS worker error: {e}", exc_info=True)
            error_event.set()
        finally:
            await audio_queue.put(_SENTINEL)

    # --- Task 3: WebSocket sender ---
    async def ws_sender():
        chunk_index = 0
        total_samples = 0
        try:
            while True:
                item = await audio_queue.get()
                if item is None or error_event.is_set():
                    break

                audio_data = _audio_to_pcm_base64(item.audio)
                total_samples += len(item.audio) if hasattr(item.audio, '__len__') else 0
                logger.info(f"WS: sending audio chunk {chunk_index} ({len(audio_data)} bytes b64)")
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_data,
                    "chunk_index": chunk_index,
                    "sentence_index": item.sentence_index,
                })
                chunk_index += 1

            total_duration = total_samples / KOKORO_SAMPLE_RATE
            gen_time = time.perf_counter() - t_start
            await websocket.send_json({
                "type": "done",
                "total_chunks": chunk_index,
                "total_duration_s": round(total_duration, 2),
                "generation_time_s": round(gen_time, 2),
            })
        except Exception as e:
            logger.error(f"WS sender error: {e}", exc_info=True)

    # Send metadata first
    await websocket.send_json({
        "type": "metadata",
        "sample_rate": KOKORO_SAMPLE_RATE,
        "encoding": "pcm_s16le",
        "channels": 1,
    })

    # Run all three tasks concurrently
    await asyncio.gather(llm_streamer(), tts_worker(), ws_sender())
