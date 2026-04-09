<p align="center">
  <h1 align="center">voz</h1>
  <p align="center">self-hosted voice AI for Brazilian Portuguese and Rioplatense Spanish</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-0.115+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/kokoro-TTS-ff6f00?logo=pytorch&logoColor=white" alt="Kokoro TTS">
  <img src="https://img.shields.io/badge/ollama-LLM-black?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  <img src="https://img.shields.io/badge/tests-41%20passing-brightgreen" alt="Tests">
</p>

---

Talk to an AI assistant in Portuguese or Spanish with regional accents. Type or speak, get a voice response back. Everything runs locally on your machine.

The core idea: stream LLM tokens through a sentence buffer into TTS, so the user hears audio before the model finishes thinking.

```
you speak/type ──> LLM (Ollama) ──> sentence buffer ──> TTS (Kokoro) ──> you hear audio
                   streaming          splits on . ? !    3.5x real-time   chunks arrive
                   tokens             fires immediately  as they're made   while LLM continues
```

## getting started

**you need:**
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with a model pulled (we use `qwen3:1.7b`)
- ~4 GB disk for model checkpoints
- Apple Silicon, NVIDIA GPU, or CPU

```bash
# install deps
uv sync

# pull the LLM
ollama pull qwen3:1.7b

# start
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run voz
```

Open **http://localhost:8000** and start talking.

## what's inside

### conversational websocket — `ws /ws/conversation`

The main thing. Send text, get back interleaved transcript + audio.

```json
// you send
{"type": "conversation", "text": "Como esta a conexao de fibra?", "language": "pt"}

// server streams back
{"type": "metadata", "sample_rate": 24000, "encoding": "pcm_s16le", "channels": 1}
{"type": "transcript", "text": "A conexao esta estavel.", "sentence_index": 0}
{"type": "audio", "data": "<base64 PCM>", "chunk_index": 0, "sentence_index": 0}
{"type": "transcript", "text": "Posso verificar mais detalhes.", "sentence_index": 1}
{"type": "audio", "data": "<base64 PCM>", "chunk_index": 1, "sentence_index": 1}
{"type": "done", "total_chunks": 2, "total_duration_s": 3.2, "generation_time_s": 1.8}
```

Three async tasks run concurrently connected by queues:
1. **LLM streamer** — streams tokens from Ollama, splits into sentences
2. **TTS worker** — synthesizes each sentence with Kokoro as soon as it's ready
3. **WS sender** — pushes audio chunks to the client

This means the user hears audio for sentence 1 while the LLM is still generating sentence 3.

### streaming TTS — `ws /ws/synthesize`

Direct text-to-speech without the LLM. Good for notifications, IVR, or testing voices.

```json
{"type": "synthesize", "text": "Bom dia, tudo bem?", "language": "pt"}
```

### REST endpoints

| Endpoint | What |
|----------|------|
| `GET /` | Chat UI |
| `GET /health` | Health check |
| `GET /voices` | Available Kokoro voices |
| `GET /accents` | Available Chatterbox accents |
| `POST /synthesize` | Chatterbox synthesis (returns WAV) |

### chat UI

The web interface at `/` lets you type or use your microphone (Web Speech API, works in Chrome). Dark theme, language/voice picker, shows transcripts and plays audio as chunks arrive.

## accents & voices

**Kokoro voices** (fast, streaming):

| Voice | Language | Gender |
|-------|----------|--------|
| `pf_dora` | Portuguese | Female |
| `pm_alex` | Portuguese | Male |
| `ef_dora` | Spanish | Female |
| `em_alex` | Spanish | Male |

**Chatterbox accents** (slow, voice cloning, offline):

| ID | Region |
|----|--------|
| `br_female` `br_male` | Generic Brazilian |
| `br_carioca` | Rio de Janeiro |
| `br_gaucho` | Rio Grande do Sul |
| `br_mineiro` | Minas Gerais |
| `br_nordestino` | Northeast |
| `br_paulista` | Sao Paulo |
| `ar_rioplatense` | Buenos Aires |

Each accent has a 10-30s reference WAV clip in `reference_audio/`.

## benchmarks (Apple M2, 8GB)

| Engine | Speed | Latency | Use case |
|--------|-------|---------|----------|
| Kokoro 82M | 3.5x real-time | ~1s | Conversational AI, streaming |
| Piper | 30x real-time | ~100ms | IVR, notifications |
| Chatterbox | 0.02x real-time | ~115s | Voice cloning (offline only) |

Run them yourself:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/benchmark_alternatives.py
```

## project structure

```
src/
  server.py           FastAPI app, REST + WebSocket endpoints
  conversation.py     pipelined LLM -> sentence buffer -> TTS -> WS
  llm_ollama.py       async Ollama streaming client
  tts_kokoro.py       Kokoro engine wrapper (streaming)
  tts_chatterbox.py   Chatterbox engine wrapper (voice cloning)
  tts_openvoice.py    OpenVoice V2 accent transfer
  pipeline.py         Chatterbox pipeline orchestration
  config.py           voices, accents, Ollama config
static/
  chat.html           web chat interface
reference_audio/      accent reference clips
scripts/              benchmarks, model download, utilities
tests/                41 tests, all ML models mocked
```

## development

```bash
uv sync --group dev
uv run pytest -v
```

## docker

```bash
docker compose up --build
```

GPU support needs NVIDIA Container Toolkit.

## device support

Auto-detects: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.

On Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before running.

## license

MIT
