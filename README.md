# TTS PoC: Regional Accent Synthesis

Proof-of-concept TTS server combining **Chatterbox Multilingual** (voice cloning)
with **OpenVoice V2** (accent transfer) for Brazilian Portuguese and Rioplatense Spanish.

Built for telecom/ISP customer service scenarios where regional accent matters.

## Architecture

Two synthesis modes:

- **chatterbox_only** (default): Direct voice cloning from reference audio. Best quality, fastest.
- **pipeline**: Chatterbox generates base speech (low `cfg_weight=0.3` to reduce accent bleed),
  then OpenVoice V2 applies accent transfer from the reference clip. Useful for cross-accent experiments.

```
POST /synthesize
     |
     v
  TTSPipeline
     |
     +-- chatterbox_only --> ChatterboxEngine --> .wav
     |
     +-- pipeline --> ChatterboxEngine --> OpenVoiceAccentConverter --> .wav
```

## Available Accents

| ID | Name | Language | Region |
|----|------|----------|--------|
| `br_female` | Brazilian Female | pt | Brazil |
| `br_male` | Brazilian Male | pt | Brazil |
| `br_carioca` | Brazilian Carioca (Rio) | pt | Brazil |
| `br_gaucho` | Brazilian Gaucho (RS) | pt | Brazil |
| `br_mineiro` | Brazilian Mineiro (MG) | pt | Brazil |
| `br_nordestino` | Brazilian Nordestino (NE) | pt | Brazil |
| `br_paulista` | Brazilian Paulista (SP) | pt | Brazil |
| `ar_rioplatense` | Rioplatense (Buenos Aires) | es | Argentina |

Each accent has a 10-30s reference WAV clip in `reference_audio/`.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ~4 GB disk for model checkpoints
- Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

### Install and Run

```bash
uv sync
uv run python scripts/download_models.py
uv run tts-server
```

Server starts at http://localhost:8000.

### OpenVoice V2 (optional, enables pipeline mode)

OpenVoice has dependency conflicts with the main project (gradio 3.x, numpy 1.22).
Install it separately:

```bash
bash scripts/install_openvoice.sh
```

Or manually:

```bash
uv pip install git+https://github.com/myshell-ai/OpenVoice.git
```

The server works without it in `chatterbox_only` mode (the default).

### Docker

```bash
docker compose up --build
```

GPU support requires NVIDIA Container Toolkit. The Docker build installs OpenVoice
automatically with a fallback if it fails.

## API Reference

### GET /health

Returns `{"status": "ok"}`.

### GET /accents

Returns list of available accent objects with `id`, `name`, `language`, `region`.

### POST /synthesize

Generate accented speech. Returns `audio/wav`.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesize (1-2000 chars) |
| `accent` | string | required | Accent ID from `/accents` |
| `mode` | string | `"chatterbox_only"` | `"chatterbox_only"` or `"pipeline"` |
| `exaggeration` | float | `0.5` | Accent emphasis (0.25-2.0) |

**Example:**

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Bom dia, tudo bem?", "accent": "br_carioca"}' \
  --output output.wav
```

## Development

```bash
uv sync --group dev
uv run pytest -v
```

## Project Structure

```
src/
  config.py           Accent registry, paths, device detection
  server.py           FastAPI endpoints
  pipeline.py         TTSPipeline orchestration
  tts_chatterbox.py   Chatterbox engine wrapper
  tts_openvoice.py    OpenVoice V2 converter wrapper
reference_audio/      10-30s WAV clips per accent
scripts/              Model download, sample generation, reference sourcing
tests/                pytest test suite (mocked ML models)
```

## Device Support

Automatic device detection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.

Apple Silicon note: `PerthImplicitWatermarker` is incompatible with MPS and is
replaced with `DummyWatermarker` at runtime.
