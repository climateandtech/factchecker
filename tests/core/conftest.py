
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch


@pytest.fixture
def mock_env(monkeypatch: MonkeyPatch) -> MonkeyPatch:
    """Fixture to set up a clean environment for each test."""
    env_vars = {
        "EMBEDDING_TYPE": None,
        "OPENAI_EMBEDDING_MODEL": None,
        "OPENAI_API_KEY": None,
        "OPENAI_API_BASE": None,
        "HUGGINGFACE_EMBEDDING_MODEL": None,
        "OLLAMA_MODEL": None,
        "OLLAMA_API_BASE_URL": None
    }
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch

@pytest.fixture
def mock_openai() -> Generator[MagicMock, None, None]:
    """Mock OpenAI embedding."""
    with patch('factchecker.core.embeddings.OpenAIEmbedding', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_huggingface() -> Generator[MagicMock, None, None]:
    """Mock HuggingFace embedding."""
    with patch('factchecker.core.embeddings.HuggingFaceEmbedding', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_ollama() -> Generator[MagicMock, None, None]:
    """Mock Ollama embedding."""
    with patch('factchecker.core.embeddings.OllamaEmbedding', autospec=True) as mock:
        yield mock

class MockEmbedding:
    """Mock embedding class for testing."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        
    def get_text_embedding(self, text: str) -> list[float]:
        """Return a mock embedding of the specified dimension."""
        return [0.0] * self.dim

    async def aget_text_embedding(self, text: str) -> list[float]:
        """Return a mock embedding of the specified dimension asynchronously."""
        return [0.0] * self.dim

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings for multiple texts."""
        return [[0.0] * self.dim for _ in texts]

    async def aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings for multiple texts asynchronously."""
        return [[0.0] * self.dim for _ in texts]

@pytest.fixture
def mock_embedding() -> MockEmbedding:
    """Fixture providing a mock embedding instance."""
    return MockEmbedding()