"""FastAPI server for TTS PoC."""

import asyncio
import base64
import logging
import time

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import ACCENTS, KOKORO_LANG_CODES, KOKORO_SAMPLE_RATE, KOKORO_VOICES
from src.pipeline import TTSPipeline, Mode
from src.tts_kokoro import KokoroEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS PoC", description="Chatterbox + OpenVoice V2 regional accent TTS")

pipeline: TTSPipeline | None = None
kokoro_engine: KokoroEngine | None = None


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


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    accent: str = Field(..., description="Accent ID (e.g., br_carioca, ar_rioplatense)")
    mode: Mode = Field("chatterbox_only", description="chatterbox_only or pipeline")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Accent emphasis")


# --- REST endpoints (Chatterbox) ---


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


def _audio_to_pcm_base64(audio: np.ndarray) -> str:
    """Convert float32 audio to base64-encoded 16-bit PCM."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


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
                await websocket.send_json({
                    "type": "audio",
                    "data": _audio_to_pcm_base64(item),
                    "chunk_index": chunk_index,
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

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


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
