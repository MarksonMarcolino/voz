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

## Streaming TTS (Kokoro)

The server includes a WebSocket endpoint for low-latency streaming synthesis using
Kokoro TTS (~3.5x real-time on Apple M2, sub-second time-to-first-audio).

### WS /ws/synthesize

Connect via WebSocket and send JSON messages:

```json
{"type": "synthesize", "text": "Bom dia, tudo bem?", "language": "pt"}
```

Optional fields: `voice` (e.g., `"pf_dora"`), `gender` (`"female"` or `"male"`).

Server streams back:
```json
{"type": "metadata", "sample_rate": 24000, "encoding": "pcm_s16le", "channels": 1}
{"type": "audio", "data": "<base64 PCM>", "chunk_index": 0}
{"type": "audio", "data": "<base64 PCM>", "chunk_index": 1}
{"type": "done", "total_chunks": 2, "total_duration_s": 3.2, "generation_time_s": 0.9}
```

The connection is persistent — send multiple requests without reconnecting.

### GET /voices

Lists available Kokoro voices with `language`, `gender`, `voice` fields.

### Benchmark Results (Apple M2, 8GB)

| Engine | Avg Speed | Use Case |
|--------|-----------|----------|
| Chatterbox | 0.02x | Voice cloning (offline only) |
| Kokoro | 3.5x | Conversational AI, streaming |
| Piper | 30x | IVR, notifications, ultra-low latency |

Run benchmarks: `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/benchmark_alternatives.py`

## REST API (Chatterbox)

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
  config.py           Accent registry, Kokoro voices, paths
  server.py           REST + WebSocket endpoints
  pipeline.py         TTSPipeline orchestration (Chatterbox)
  tts_kokoro.py       Kokoro engine wrapper (streaming)
  tts_chatterbox.py   Chatterbox engine wrapper (voice cloning)
  tts_openvoice.py    OpenVoice V2 converter wrapper
reference_audio/      10-30s WAV clips per accent
scripts/              Benchmarks, model download, sample generation
tests/                pytest test suite (28 tests, mocked ML models)
```

## Device Support

Automatic device detection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.

Apple Silicon note: `PerthImplicitWatermarker` is incompatible with MPS and is
replaced with `DummyWatermarker` at runtime.
