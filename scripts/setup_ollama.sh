#!/usr/bin/env bash
# Pull the LLMs evaluated in the Compass paper.
# Run once after installing Ollama (https://ollama.com).
#
# About 30 GB total disk; pulls run in parallel where possible.

set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found. Install from https://ollama.com first." >&2
    exit 1
fi

MODELS=(
    "llama3.2:1b"
    "llama3.2:3b"
    "llama3.1:8b"
    "gemma3:1b"
    "gemma3:4b"
    "gemma3:12b"
)

for model in "${MODELS[@]}"; do
    echo ">>> ollama pull $model"
    ollama pull "$model"
done

echo "All models pulled. Listing:"
ollama list
