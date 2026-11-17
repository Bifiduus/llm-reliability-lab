#!/usr/bin/env bash
set -euo pipefail

# Move to project root
cd "$(dirname "$0")/.."

echo "=== LLM Reliability Lab Setup ==="

# 1. Python environment
echo "Creating Python environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Python dependencies
echo "Installing Python dependencies..."
python -m pip install -e ".[dev]"

# 3. Download corpus
echo "Downloading documents..."
python scripts/download_documents.py

# 4. Generate chunks
echo "Generating chunks..."
python src/export_chunks.py

# 5. Download retrieval models
echo "Downloading retrieval models..."

python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Retrieval models ready.")
PY

# 6. Install Ollama
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y zstd

    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 7. Check Ollama server
echo "Checking Ollama server..."

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve > /tmp/llm-reliability-ollama.log 2>&1 &
fi

# Wait for Ollama to become available
ready=false

for i in $(seq 1 30); do
    if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 2
done

if [ "$ready" != "true" ]; then
    echo "Ollama server could not be started."
    echo "Try running: ollama serve"
    exit 1
fi

# 8. Download LLM
echo "Downloading Qwen2.5..."
ollama pull qwen2.5:3b

echo ""
echo "=== Setup completed successfully ==="
echo "Activate environment: source .venv/bin/activate"
echo "Run retrieval: python src/evaluate.py"
echo "Run RAG: python src/evaluate_rag.py"
