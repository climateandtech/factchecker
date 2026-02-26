#!/bin/bash
# Run a fact-checking experiment using the project venv and .env (e.g. Ollama).
# Usage: ./run_with_venv.sh [--experiment MODULE] [options...]
# Example: ./run_with_venv.sh --experiment factchecker.experiments.ragatouille_colbert_ir
# Default experiment: advocate-mediator climatefeedback.

set -e
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON=""
for p in python3.12 python3.10; do
  if command -v "$p" &>/dev/null; then
    PYTHON="$p"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "Need Python 3.12 or 3.10. Install with e.g. pyenv or your system package manager."
  exit 1
fi
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ($PYTHON)..."
  "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if ! python -c "import llama_index" 2>/dev/null; then
  echo "Installing dependencies (pip install -e .)..."
  pip install -e .
fi

if [[ ! -f .env ]]; then
  echo "No .env found. Copy .env.example to .env and set LLM_TYPE=ollama, EMBEDDING_TYPE=ollama for Ollama."
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Edit .env and run again."
    exit 1
  fi
  exit 1
fi

python run_experiments.py "$@"
