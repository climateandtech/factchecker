from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from factchecker.core.embeddings import load_embedding_model


def test_load_openai_embedding_default(mock_env: MonkeyPatch, mock_openai: MagicMock) -> None:
    """
    Test loading OpenAI embedding with default settings.

    :param mock_env: Fixture to manipulate environment variables.
    :param mock_openai: Mocked OpenAI embedding instance.
    """
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    
    _ = load_embedding_model()
    
    mock_openai.assert_called_once_with(
        model_name="text-embedding-ada-002",
        api_key="test-key",
        api_base=None
    )

def test_load_openai_embedding_custom(mock_openai: MagicMock) -> None:
    """
    Test loading OpenAI embedding with custom settings.

    :param mock_openai: Mocked OpenAI embedding instance.
    """
    _ = load_embedding_model(
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

def test_load_openai_embedding_missing_api_key(mock_env: MonkeyPatch) -> None:
    """
    Test error handling when OpenAI API key is missing.

    :param mock_env: Fixture to manipulate environment variables.
    """
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        load_embedding_model(embedding_type="openai")

def test_openai_embedding_type_fallback(mock_env: MonkeyPatch, mock_openai: MagicMock) -> None:
    """Test fallback to OpenAI when no embedding type is specified."""
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    
    _ = load_embedding_model()
    
    mock_openai.assert_called_once_with(
        model_name="text-embedding-ada-002",
        api_key="test-key",
        api_base=None
    )

def test_load_openai_embedding_with_extra_kwargs(mock_env: MonkeyPatch, mock_openai: MagicMock) -> None:
    """Test passing additional kwargs to embedding models."""
    extra_kwargs = {"extra_param": "value"}
    
    _ = load_embedding_model(
        model_name="text-embedding-ada-002",
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

def test_load_huggingface_embedding(mock_env: MonkeyPatch, mock_huggingface: MagicMock) -> None:
    """Test loading HuggingFace embedding."""
    _ = load_embedding_model(
        embedding_type="huggingface",
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    mock_huggingface.assert_called_once_with(
        model_name="BAAI/bge-small-en-v1.5"
    )

def test_load_huggingface_embedding_from_env(mock_env: MonkeyPatch, mock_huggingface: MagicMock) -> None:
    """Test loading HuggingFace embedding."""
    mock_env.setenv("EMBEDDING_TYPE", "huggingface")
    mock_env.setenv("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    _ = load_embedding_model()
    
    mock_huggingface.assert_called_once_with(
        model_name="BAAI/bge-small-en-v1.5"
    )

def test_load_huggingface_embedding_custom(mock_huggingface: MagicMock) -> None:
    """Test loading HuggingFace embedding with custom settings."""
    extra_kwargs = {"device": "cpu", "normalize_embeddings": True}
    
    _ = load_embedding_model(
        embedding_type="huggingface",
        model_name="custom/model",
        **extra_kwargs
    )
    
    mock_huggingface.assert_called_once_with(
        model_name="custom/model",
        **extra_kwargs
    )

def test_load_ollama_embedding(mock_env: MonkeyPatch, mock_ollama: MagicMock) -> None:
    """Test loading Ollama embedding."""
    _ = load_embedding_model(
        embedding_type="ollama",
        model_name="nomic-embed-text",
        api_base="http://localhost:11434",
    )
    
    mock_ollama.assert_called_once_with(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

def test_load_ollama_embedding_from_env(mock_env: MonkeyPatch, mock_ollama: MagicMock) -> None:
    """Test loading Ollama embedding."""
    mock_env.setenv("EMBEDDING_TYPE", "ollama")
    mock_env.setenv("OLLAMA_MODEL", "nomic-embed-text")
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    _ = load_embedding_model()
    
    mock_ollama.assert_called_once_with(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )


def test_load_ollama_embedding_custom(mock_ollama: MagicMock) -> None:
    """Test loading Ollama embedding with custom settings."""
    _ = load_embedding_model(
        embedding_type="ollama",
        model_name="custom-model",
        api_base="http://custom-server:11434"
    )
    
    mock_ollama.assert_called_once_with(
        model_name="custom-model",
        base_url="http://custom-server:11434"
    )

def test_load_ollama_embedding_with_kwargs(mock_ollama: MagicMock) -> None:
    """Test loading Ollama embedding with extra kwargs."""
    extra_kwargs = {"timeout": 30, "request_timeout": 60}
    
    _ = load_embedding_model(
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


def test_invalid_embedding_type() -> None:
    """Test error handling for invalid embedding type."""
    with pytest.raises(ValueError, match="Unsupported embedding type: invalid"):
        load_embedding_model(embedding_type="invalid")