"""Voz — voice conversational AI server."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from src.config import ACCENTS, KOKORO_LANG_CODES, KOKORO_SAMPLE_RATE, KOKORO_VOICES
from src.conversation import run_conversation
from src.history import ConversationHistory
from src.pipeline import TTSPipeline, Mode
from src.stt_whisper import WhisperEngine, build_stt_prompt
from src.tts_kokoro import KokoroEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm Whisper + Kokoro in background threads so the first hands-free
    # utterance and the first TTS response don't pay 20-30s of model load time.
    async def _warm():
        try:
            logger.info("Warming Whisper model in background...")
            stt = get_whisper_engine()
            executor = getattr(stt, "_executor", None)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, stt._get_model)
            logger.info("Whisper ready")
        except Exception as e:
            logger.warning(f"Whisper warm failed: {e}")
        for lang_code in ("p", "e"):
            try:
                logger.info(f"Warming Kokoro pipeline {lang_code!r} in background...")
                await asyncio.to_thread(get_kokoro_engine()._get_pipeline, lang_code)
                logger.info(f"Kokoro {lang_code!r} ready")
            except Exception as e:
                logger.warning(f"Kokoro {lang_code!r} warm failed: {e}")

    asyncio.create_task(_warm())
    yield


app = FastAPI(
    title="Voz",
    description="Self-hosted voice conversational AI for BR-PT and AR-ES",
    lifespan=lifespan,
)


STATIC_DIR = Path(__file__).parent.parent / "static"
VENDOR_DIR = STATIC_DIR / "vendor"

# Hands-free VAD assets are vendored locally so the browser loads them
# same-origin — cross-origin ESM imports of ort's WASM glue fail in Safari
# and other browsers, breaking @ricky0123/vad-web init.
_VENDOR_ASSETS = {
    # vad-web bundle MUST be same-origin: its dynamic import() of the ort .mjs
    # uses the calling script's base URL, which becomes about:blank for
    # cross-origin scripts and breaks resolution of /static/vendor/... paths.
    "vad/bundle.min.js":
        "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js",
    "vad/silero_vad_legacy.onnx":
        "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/silero_vad_legacy.onnx",
    "vad/silero_vad_v5.onnx":
        "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/silero_vad_v5.onnx",
    "vad/vad.worklet.bundle.min.js":
        "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/vad.worklet.bundle.min.js",
    # vad-web bundles ort@1.22 internally; it loads these via dynamic import
    "ort/ort-wasm-simd-threaded.wasm":
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.wasm",
    "ort/ort-wasm-simd-threaded.mjs":
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.mjs",
}


def _ensure_vendor_assets() -> None:
    import urllib.request
    for rel, url in _VENDOR_ASSETS.items():
        dest = VENDOR_DIR / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading vendor asset: {rel}")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                dest.write_bytes(resp.read())
            logger.info(f"  saved {rel} ({dest.stat().st_size} bytes)")
        except Exception as e:
            logger.warning(f"  failed to fetch {rel}: {e}")


_ensure_vendor_assets()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline: TTSPipeline | None = None
kokoro_engine: KokoroEngine | None = None
whisper_engine: WhisperEngine | None = None


def get_pipeline() -> TTSPipeline:
    global pipeline
    if pipeline is None:
        pipeline = TTSPipeline()
    return pipeline


def get_kokoro_engine() -> KokoroEngine:
    global kokoro_engine
    if kokoro_engine is None:
        kokoro_engine = KokoroEngine()
    return kokoro_engine


def get_whisper_engine():
    """Pick the fastest Whisper backend for the current platform.

    Apple Silicon → mlx-whisper (~5-7x faster than CTranslate2 on Mac).
    Everything else → faster-whisper int8 on CPU.
    """
    global whisper_engine
    if whisper_engine is None:
        import platform
        import sys
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            from src.stt_mlx import MlxWhisperEngine
            whisper_engine = MlxWhisperEngine()
            logger.info("STT backend: mlx-whisper (Apple Silicon)")
        else:
            whisper_engine = WhisperEngine()
            logger.info("STT backend: faster-whisper (CTranslate2)")
    return whisper_engine


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    accent: str = Field(..., description="Accent ID (e.g., br_carioca, ar_rioplatense)")
    mode: Mode = Field("chatterbox_only", description="chatterbox_only or pipeline")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Accent emphasis")


# --- REST endpoints (Chatterbox) ---


@app.get("/")
def chat_ui():
    """Serve the chat interface."""
    return HTMLResponse((STATIC_DIR / "chat.html").read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/accents")
def list_accents():
    return TTSPipeline.list_accents()


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if req.accent not in ACCENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown accent: {req.accent}. Available: {list(ACCENTS.keys())}",
        )

    try:
        p = get_pipeline()
        output = p.synthesize(
            text=req.text,
            accent_id=req.accent,
            mode=req.mode,
            exaggeration=req.exaggeration,
        )
        return FileResponse(
            str(output),
            media_type="audio/wav",
            filename=f"{req.accent}_{req.mode}.wav",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- WebSocket endpoint (Kokoro streaming) ---


def _audio_to_pcm_bytes(audio) -> bytes:
    """Convert float32 audio (numpy or torch tensor) to 16-bit PCM bytes."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


@app.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket):
    await websocket.accept()
    engine = get_kokoro_engine()

    try:
        while True:
            data = await websocket.receive_json()

            msg_type = data.get("type")
            if msg_type != "synthesize":
                await websocket.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})
                continue

            text = data.get("text", "").strip()
            if not text:
                await websocket.send_json({"type": "error", "detail": "Empty text"})
                continue

            if len(text) > 2000:
                await websocket.send_json({"type": "error", "detail": "Text exceeds 2000 characters"})
                continue

            language = data.get("language", "pt")
            if language not in KOKORO_LANG_CODES:
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Unknown language: {language}. Available: {list(KOKORO_LANG_CODES.keys())}",
                })
                continue

            voice = data.get("voice")
            gender = data.get("gender", "female")

            # Send metadata before audio starts
            await websocket.send_json({
                "type": "metadata",
                "sample_rate": KOKORO_SAMPLE_RATE,
                "encoding": "pcm_s16le",
                "channels": 1,
            })

            # Stream Kokoro chunks from a thread (Kokoro is sync/CPU-bound)
            chunk_index = 0
            total_samples = 0
            t_start = time.perf_counter()

            queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _generate():
                try:
                    for chunk in engine.stream(text, language, voice, gender):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, _generate)

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    await websocket.send_json({"type": "error", "detail": str(item)})
                    break

                total_samples += len(item)
                await websocket.send_bytes(_audio_to_pcm_bytes(item))
                chunk_index += 1

            total_duration = total_samples / KOKORO_SAMPLE_RATE
            gen_time = time.perf_counter() - t_start

            await websocket.send_json({
                "type": "done",
                "total_chunks": chunk_index,
                "total_duration_s": round(total_duration, 2),
                "generation_time_s": round(gen_time, 2),
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# --- Conversational WebSocket (LLM → TTS pipeline) ---


@app.websocket("/ws/conversation")
async def ws_conversation(websocket: WebSocket):
    await websocket.accept()
    engine = get_kokoro_engine()
    history = ConversationHistory()
    conversation_task: asyncio.Task | None = None
    interrupt_event: asyncio.Event | None = None

    async def _interrupt_active():
        """Interrupt the current conversation task if running."""
        nonlocal conversation_task, interrupt_event
        if conversation_task and not conversation_task.done():
            interrupt_event.set()
            await conversation_task
        conversation_task = None
        interrupt_event = None

    # Track current language/voice for audio input (STT) context
    current_language = "pt"
    current_voice = None
    current_gender = "female"

    async def _start_conversation(data: dict):
        """Validate input and start a new conversation task."""
        nonlocal conversation_task, interrupt_event, current_language, current_voice, current_gender

        text = data.get("text", "").strip()
        if not text:
            await websocket.send_json({"type": "error", "detail": "Empty text"})
            return

        language = data.get("language", "pt")
        if language not in KOKORO_LANG_CODES:
            await websocket.send_json({
                "type": "error",
                "detail": f"Unknown language: {language}. Available: {list(KOKORO_LANG_CODES.keys())}",
            })
            return

        voice = data.get("voice")
        gender = data.get("gender", "female")

        # Track for audio input fallback
        current_language = language
        current_voice = voice
        current_gender = gender

        interrupt_event = asyncio.Event()
        conversation_task = asyncio.create_task(
            run_conversation(
                websocket=websocket,
                text=text,
                language=language,
                engine=engine,
                voice=voice,
                gender=gender,
                history=history,
                interrupt_event=interrupt_event,
            )
        )

    async def _handle_audio_input(audio_bytes: bytes):
        """Transcribe audio input and start a conversation."""
        loop = asyncio.get_event_loop()
        stt = get_whisper_engine()

        # Bias Whisper toward the bot's prior turn + static glossary; this
        # massively improves recognition of proper nouns and domain terms.
        prompt = build_stt_prompt(history, current_language)
        # MLX engines pin transcribe to a single dedicated worker (MLX has
        # per-thread GPU command queues; thread hopping costs 5-15s/call).
        executor = getattr(stt, "_executor", None)
        text = await loop.run_in_executor(
            executor, stt.transcribe, audio_bytes, current_language, prompt
        )

        # User may have refreshed or navigated away during a slow first STT call.
        # Don't run the LLM/TTS pipeline into a closed socket.
        if websocket.client_state != WebSocketState.CONNECTED:
            logger.info(f"WS closed during STT, dropping transcript: {text!r}")
            return

        if not text.strip():
            await websocket.send_json({"type": "error", "detail": "Could not transcribe audio"})
            return

        await websocket.send_json({"type": "transcript_input", "text": text})

        await _start_conversation({
            "text": text,
            "language": current_language,
            "voice": current_voice,
            "gender": current_gender,
        })

    try:
        while True:
            message = await websocket.receive()

            # Binary frame = audio input for STT
            if "bytes" in message and message["bytes"]:
                await _interrupt_active()
                await _handle_audio_input(message["bytes"])
                continue

            # Text frame = JSON control message
            data = json.loads(message.get("text", "{}"))
            msg_type = data.get("type")

            if msg_type == "interrupt":
                await _interrupt_active()

            elif msg_type == "clear_history":
                history.clear()
                await websocket.send_json({"type": "history_cleared"})

            elif msg_type == "conversation":
                # Auto-interrupt if already responding
                await _interrupt_active()
                await _start_conversation(data)

            elif msg_type == "set_language":
                current_language = data.get("language", current_language)
                current_voice = data.get("voice", current_voice)
                current_gender = data.get("gender", current_gender)

            else:
                await websocket.send_json({"type": "error", "detail": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Conversation WebSocket disconnected")
        if conversation_task and not conversation_task.done():
            conversation_task.cancel()
    except Exception as e:
        logger.error(f"Conversation WebSocket error: {e}", exc_info=True)


# --- Kokoro voices REST endpoint ---


@app.get("/voices")
def list_voices():
    """List available Kokoro voices for streaming synthesis."""
    result = []
    for lang, voices in KOKORO_VOICES.items():
        for gender, voice_id in voices.items():
            result.append({"language": lang, "gender": gender, "voice": voice_id})
    return result


def main():
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
