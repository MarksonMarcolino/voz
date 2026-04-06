#!/usr/bin/env bash
set -euo pipefail

# OpenVoice V2 has conflicting transitive dependencies (gradio 3.x, numpy 1.22)
# that prevent it from being a normal pyproject.toml dependency.
# This script installs it separately into the existing venv.

echo "Installing OpenVoice V2 (separate install due to dependency conflicts)..."
echo "This may adjust numpy version. The TTS server handles both versions."
echo ""

uv pip install git+https://github.com/myshell-ai/OpenVoice.git

echo ""
echo "Done. 'pipeline' mode is now available in the TTS server."
