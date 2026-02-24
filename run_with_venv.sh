#!/bin/bash
# Run the advocate-mediator experiment using the project venv and .env for Ollama.
# Ensure .env has LLM_TYPE=ollama, EMBEDDING_TYPE=ollama, OLLAMA_API_BASE_URL, etc.

set -e
cd "$(dirname "$0")"

# Use .venv in project root
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR (Python 3.12)..."
  python3.12 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install deps if needed
if ! python -c "import llama_index" 2>/dev/null; then
  echo "Installing dependencies (pip install -e .)..."
  pip install -e .
fi

# .env is loaded by the experiment module; copy from example if missing
if [[ ! -f .env ]]; then
  echo "No .env found. Copy .env.example to .env and set LLM_TYPE=ollama, EMBEDDING_TYPE=ollama for Ollama."
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Edit .env and run again."
    exit 1
  fi
  exit 1
fi

echo "Running experiment (Ollama settings from .env)..."
python -m factchecker.experiments.advocate_mediator_climatefeedback.advocate_mediator_climatefeedback "$@"
