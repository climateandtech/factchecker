#!/bin/bash
set -e

# Deactivate any existing venv
deactivate 2>/dev/null || true

# Remove existing venv
rm -rf venv

# Prefer Python 3.12, fallback to 3.10
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
echo "Using $PYTHON for venv"
"$PYTHON" -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade basic packages
python -m pip install --upgrade pip setuptools wheel

# Install package in editable mode with test dependencies
python -m pip install -e ".[test]" --verbose --use-pep517

# Run tests with coverage
export IS_TESTING=true
export LLAMA_INDEX_EMBED_MODEL=mock
export MOCK_EMBED_DIM=8
python -m pytest tests/ --cov=factchecker --cov-report=xml -v 