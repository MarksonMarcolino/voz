FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/
COPY scripts/ scripts/

# OpenVoice V2 (separate install due to dependency conflicts)
# Falls back gracefully -- server runs in chatterbox_only mode if this fails
RUN uv pip install git+https://github.com/myshell-ai/OpenVoice.git || \
    echo "WARNING: OpenVoice install failed. Only chatterbox_only mode available."

# Download models at build time
RUN uv run python scripts/download_models.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
