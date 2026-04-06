"""FastAPI server for TTS PoC."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import ACCENTS
from src.pipeline import TTSPipeline, Mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS PoC", description="Chatterbox + OpenVoice V2 regional accent TTS")

pipeline: TTSPipeline | None = None


def get_pipeline() -> TTSPipeline:
    global pipeline
    if pipeline is None:
        pipeline = TTSPipeline()
    return pipeline


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    accent: str = Field(..., description="Accent ID (e.g., br_carioca, ar_rioplatense)")
    mode: Mode = Field("chatterbox_only", description="chatterbox_only or pipeline")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Accent emphasis")


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


def main():
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
