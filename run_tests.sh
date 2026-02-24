#!/bin/bash
set -e

# Deactivate any existing venv
deactivate 2>/dev/null || true

# Remove existing venv
rm -rf venv

# Create fresh venv with Python 3.12
python3.12 -m venv venv

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