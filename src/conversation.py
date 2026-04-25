"""Pipelined conversation: LLM streaming → sentence buffer → TTS streaming → WebSocket."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from fastapi import WebSocket

from src.config import KOKORO_SAMPLE_RATE, SENTENCE_MIN_LENGTH, SYSTEM_PROMPTS
from src.llm_ollama import OllamaError, stream_chat
from src.tts_kokoro import KokoroEngine

if TYPE_CHECKING:
    from src.history import ConversationHistory

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_MD_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_MD_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)|(?<![_\w])_([^_\n]+)_(?!_)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_LIST_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_MD_LIST_NUMBER = re.compile(r"^\s{0,3}\d+\.\s+", re.MULTILINE)


def _strip_markdown_for_tts(text: str) -> str:
    """Strip markdown formatting so Kokoro doesn't vocalize asterisks/backticks/etc."""
    text = _MD_CODE_FENCE.sub("", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_ITALIC.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_HEADING.sub("", text)
    text = _MD_LIST_BULLET.sub("", text)
    text = _MD_LIST_NUMBER.sub("", text)
    return text.strip()


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


def _audio_to_pcm_bytes(audio) -> bytes:
    """Convert float32 audio (numpy or torch tensor) to 16-bit PCM bytes."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


_SENTINEL = None


async def run_conversation(
    websocket: WebSocket,
    text: str,
    language: str,
    engine: KokoroEngine,
    voice: str | None = None,
    gender: str = "female",
    history: ConversationHistory | None = None,
    interrupt_event: asyncio.Event | None = None,
) -> None:
    """Run the full LLM → TTS → WebSocket pipeline with streaming overlap."""

    sentence_queue: asyncio.Queue[_SentenceReady | None] = asyncio.Queue()
    audio_queue: asyncio.Queue[_AudioChunk | None] = asyncio.Queue()
    error_event = asyncio.Event()
    if interrupt_event is None:
        interrupt_event = asyncio.Event()
    t_start = time.perf_counter()
    timings = {"llm_first_token": None, "llm_done": None, "first_audio_sent": None}
    full_response_parts: list[str] = []

    def _should_stop() -> bool:
        return error_event.is_set() or interrupt_event.is_set()

    # Add user message to history before LLM call
    if history is not None:
        history.add_user(text)

    # Build messages list from history or fall back to single-turn
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["pt"])
    llm_messages = history.get_messages(system_prompt) if history is not None else None

    # --- Task 1: LLM streaming + sentence splitting ---
    async def llm_streamer():
        try:
            buffer = SentenceBuffer()
            sentence_index = 0

            async for token in stream_chat(text, language, messages=llm_messages):
                if _should_stop():
                    return
                if timings["llm_first_token"] is None:
                    timings["llm_first_token"] = time.perf_counter() - t_start
                sentences = buffer.add(token)
                for s in sentences:
                    logger.info(f"LLM: sentence {sentence_index} ready: \"{s[:60]}\"")
                    full_response_parts.append(s)
                    await sentence_queue.put(_SentenceReady(text=s, index=sentence_index))
                    sentence_index += 1

            remaining = buffer.flush()
            if remaining:
                logger.info(f"LLM: final sentence {sentence_index}: \"{remaining[:60]}\"")
                full_response_parts.append(remaining)
                await sentence_queue.put(_SentenceReady(text=remaining, index=sentence_index))

            timings["llm_done"] = time.perf_counter() - t_start

            # Save assistant response to history
            if history is not None and full_response_parts:
                history.add_assistant(" ".join(full_response_parts))

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
                if item is None or _should_stop():
                    break

                sentence = item
                logger.info(f"TTS [{sentence.index}]: synthesizing \"{sentence.text[:60]}\"")

                if _should_stop():
                    break

                # Send transcript before audio
                await websocket.send_json({
                    "type": "transcript",
                    "text": sentence.text,
                    "sentence_index": sentence.index,
                })

                # Run Kokoro in thread pool with timeout
                text_to_speak = _strip_markdown_for_tts(sentence.text)
                if not text_to_speak:
                    logger.info(f"TTS [{sentence.index}]: skipped (empty after markdown strip)")
                    continue
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

                if _should_stop():
                    break

                if not chunks:
                    logger.warning(f"TTS [{sentence.index}]: no audio generated")
                    continue

                for chunk in chunks:
                    arr = chunk if isinstance(chunk, np.ndarray) else np.asarray(chunk, dtype=np.float32)
                    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
                    logger.info(
                        f"TTS [{sentence.index}]: audio chunk, {arr.size} samples, peak={peak:.3f}"
                    )
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
                if item is None or _should_stop():
                    break

                pcm_bytes = _audio_to_pcm_bytes(item.audio)
                total_samples += len(item.audio) if hasattr(item.audio, '__len__') else 0
                if timings["first_audio_sent"] is None:
                    timings["first_audio_sent"] = time.perf_counter() - t_start
                logger.info(f"WS: sending audio chunk {chunk_index} ({len(pcm_bytes)} bytes)")
                await websocket.send_bytes(pcm_bytes)
                chunk_index += 1

            total_duration = total_samples / KOKORO_SAMPLE_RATE
            gen_time = time.perf_counter() - t_start

            if interrupt_event.is_set():
                await websocket.send_json({"type": "interrupted"})
                # Still save partial response to history
                if history is not None and full_response_parts:
                    history.add_assistant(" ".join(full_response_parts))
            else:
                await websocket.send_json({
                    "type": "done",
                    "total_chunks": chunk_index,
                    "total_duration_s": round(total_duration, 2),
                    "generation_time_s": round(gen_time, 2),
                    "llm_first_token_s": round(timings["llm_first_token"], 2) if timings["llm_first_token"] else None,
                    "llm_total_s": round(timings["llm_done"], 2) if timings["llm_done"] else None,
                    "first_audio_s": round(timings["first_audio_sent"], 2) if timings["first_audio_sent"] else None,
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
