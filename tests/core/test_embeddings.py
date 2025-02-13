import pytest
from unittest.mock import patch, Mock
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from factchecker.core.embeddings import load_embedding_model

class MockEmbedding:
    """Mock embedding class for testing."""
    def __init__(self, dim=384):
        self.dim = dim
        
    def get_text_embedding(self, text):
        """Return a mock embedding of the specified dimension."""
        return [0.0] * self.dim

    async def aget_text_embedding(self, text):
        """Return a mock embedding of the specified dimension asynchronously."""
        return [0.0] * self.dim

    def get_text_embeddings(self, texts):
        """Return mock embeddings for multiple texts."""
        return [[0.0] * self.dim for _ in texts]

    async def aget_text_embeddings(self, texts):
        """Return mock embeddings for multiple texts asynchronously."""
        return [[0.0] * self.dim for _ in texts]

@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to set up a clean environment for each test"""
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
def mock_openai():
    """Mock OpenAI embedding"""
    with patch('factchecker.core.embeddings.OpenAIEmbedding', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_huggingface():
    """Mock HuggingFace embedding"""
    with patch('factchecker.core.embeddings.HuggingFaceEmbedding', autospec=True) as mock:
        yield mock

@pytest.fixture
def mock_ollama():
    """Mock Ollama embedding"""
    with patch('factchecker.core.embeddings.OllamaEmbedding', autospec=True) as mock:
        yield mock

def test_load_openai_embedding_default(mock_env, mock_openai):
    """Test loading OpenAI embedding with default settings"""
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    
    embedding = load_embedding_model()
    
    mock_openai.assert_called_once_with(
        model_name="text-embedding-ada-002",
        api_key="test-key",
        api_base=None
    )

def test_load_openai_embedding_custom(mock_openai):
    """Test loading OpenAI embedding with custom settings"""
    embedding = load_embedding_model(
        embedding_type="openai",
        model_name="custom-embedding-model",
        api_key="custom-key",
        api_base="custom-base"
    )
    
    mock_openai.assert_called_once_with(
        model_name="custom-embedding-model",
        api_key="custom-key",
        api_base="custom-base"
    )

def test_load_openai_embedding_missing_api_key(mock_env):
    """Test error when OpenAI API key is missing"""
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        load_embedding_model(embedding_type="openai")

def test_load_huggingface_embedding(mock_env, mock_huggingface):
    """Test loading HuggingFace embedding"""
    mock_env.setenv("EMBEDDING_TYPE", "huggingface")
    mock_env.setenv("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    embedding = load_embedding_model()
    
    mock_huggingface.assert_called_once_with(
        model_name="BAAI/bge-small-en-v1.5"
    )

def test_load_huggingface_embedding_custom(mock_huggingface):
    """Test loading HuggingFace embedding with custom settings"""
    extra_kwargs = {"device": "cpu", "normalize_embeddings": True}
    
    embedding = load_embedding_model(
        embedding_type="huggingface",
        model_name="custom/model",
        **extra_kwargs
    )
    
    mock_huggingface.assert_called_once_with(
        model_name="custom/model",
        **extra_kwargs
    )

def test_load_ollama_embedding(mock_env, mock_ollama):
    """Test loading Ollama embedding"""
    mock_env.setenv("EMBEDDING_TYPE", "ollama")
    mock_env.setenv("OLLAMA_MODEL", "nomic-embed-text")
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    embedding = load_embedding_model()
    
    mock_ollama.assert_called_once_with(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

def test_load_ollama_embedding_custom(mock_ollama):
    """Test loading Ollama embedding with custom settings"""
    embedding = load_embedding_model(
        embedding_type="ollama",
        model_name="custom-model",
        api_base="http://custom-server:11434"
    )
    
    mock_ollama.assert_called_once_with(
        model_name="custom-model",
        base_url="http://custom-server:11434"
    )

def test_load_ollama_embedding_with_kwargs(mock_ollama):
    """Test loading Ollama embedding with extra kwargs"""
    extra_kwargs = {"timeout": 30, "request_timeout": 60}
    
    embedding = load_embedding_model(
        embedding_type="ollama",
        model_name="custom-model",
        api_base="http://custom-server:11434",
        **extra_kwargs
    )
    
    mock_ollama.assert_called_once_with(
        model_name="custom-model",
        base_url="http://custom-server:11434",
        **extra_kwargs
    )

def test_embedding_type_fallback(mock_env, mock_openai):
    """Test fallback to OpenAI when no embedding type is specified"""
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    
    embedding = load_embedding_model()
    
    mock_openai.assert_called_once_with(
        model_name="text-embedding-ada-002",
        api_key="test-key",
        api_base=None
    )

def test_load_embedding_with_extra_kwargs(mock_openai):
    """Test passing additional kwargs to embedding models"""
    extra_kwargs = {"extra_param": "value"}
    
    embedding = load_embedding_model(
        embedding_type="openai",
        api_key="test-key",
        **extra_kwargs
    )
    
    mock_openai.assert_called_once_with(
        model_name="text-embedding-ada-002",
        api_key="test-key",
        api_base=None,
        **extra_kwargs
    )

def test_invalid_embedding_type():
    """Test error handling for invalid embedding type"""
    with pytest.raises(ValueError, match="Unsupported embedding type: invalid"):
        load_embedding_model(embedding_type="invalid") 