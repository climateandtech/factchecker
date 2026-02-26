from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

from factchecker.core.embeddings import load_embedding_model
from factchecker.core.ollama_batch_embedding import BatchedOllamaEmbedding


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
    """Test loading Ollama embedding from env (OLLAMA_MODEL)."""
    mock_env.setenv("EMBEDDING_TYPE", "ollama")
    mock_env.setenv("OLLAMA_MODEL", "nomic-embed-text")
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    _ = load_embedding_model()
    
    mock_ollama.assert_called_once_with(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )


def test_load_ollama_embedding_uses_embedding_model_env(mock_env: MonkeyPatch, mock_ollama: MagicMock) -> None:
    """When OLLAMA_EMBEDDING_MODEL is set, it is used for embeddings (overrides OLLAMA_MODEL)."""
    mock_env.setenv("EMBEDDING_TYPE", "ollama")
    mock_env.setenv("OLLAMA_EMBEDDING_MODEL", "jina/jina-embeddings-v2-base-de")
    mock_env.setenv("OLLAMA_MODEL", "llama3.2:latest")
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    _ = load_embedding_model()
    
    mock_ollama.assert_called_once_with(
        model_name="jina/jina-embeddings-v2-base-de",
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


def test_ollama_batch_uses_embed_api() -> None:
    """BatchedOllamaEmbedding._get_text_embeddings calls client.embed(input=list), not per-text embeddings()."""
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.0] * 8, [0.0] * 8]}
    with patch("llama_index.embeddings.ollama.base.Client", return_value=mock_client), patch(
        "llama_index.embeddings.ollama.base.AsyncClient", return_value=MagicMock()
    ):
        emb = BatchedOllamaEmbedding(
            model_name="test-model",
            base_url="http://localhost:11434",
            embed_batch_size=10,
        )
        emb._client = mock_client
    result = emb._get_text_embeddings(["text one", "text two"])
    assert len(result) == 2
    mock_client.embed.assert_called_once()
    call_kw = mock_client.embed.call_args[1]
    assert call_kw["input"] == ["text one", "text two"]
    assert call_kw["model"] == "test-model"


def test_ollama_batch_respects_embed_batch_size() -> None:
    """BatchedOllamaEmbedding chunks by embed_batch_size and calls embed per chunk."""
    mock_client = MagicMock()
    # Return 2, 2, 1 vectors for the three chunked calls
    mock_client.embed.side_effect = [
        {"embeddings": [[0.0] * 8, [0.0] * 8]},
        {"embeddings": [[0.0] * 8, [0.0] * 8]},
        {"embeddings": [[0.0] * 8]},
    ]
    with patch("llama_index.embeddings.ollama.base.Client", return_value=mock_client), patch(
        "llama_index.embeddings.ollama.base.AsyncClient", return_value=MagicMock()
    ):
        emb = BatchedOllamaEmbedding(
            model_name="m",
            base_url="http://localhost:11434",
            embed_batch_size=2,
        )
        emb._client = mock_client
    # 5 texts, batch_size=2 -> 3 calls: [0:2], [2:4], [4:5]
    result = emb._get_text_embeddings(["a", "b", "c", "d", "e"])
    assert len(result) == 5
    assert mock_client.embed.call_count == 3
    calls = [c[1]["input"] for c in mock_client.embed.call_args_list]
    assert calls == [["a", "b"], ["c", "d"], ["e"]]


def test_ollama_batch_returns_correct_shape() -> None:
    """BatchedOllamaEmbedding returns one vector per input text."""
    dim = 8
    mock_client = MagicMock()
    mock_client.embed.return_value = {
        "embeddings": [[0.1] * dim, [0.2] * dim],
    }
    with patch("llama_index.embeddings.ollama.base.Client", return_value=mock_client), patch(
        "llama_index.embeddings.ollama.base.AsyncClient", return_value=MagicMock()
    ):
        emb = BatchedOllamaEmbedding(
            model_name="m",
            base_url="http://localhost:11434",
        )
        emb._client = mock_client
    result = emb._get_text_embeddings(["a", "b"])
    assert len(result) == 2
    assert len(result[0]) == dim and len(result[1]) == dim
    assert result[0][0] == 0.1 and result[1][0] == 0.2


@pytest.mark.integration
def test_ollama_batch_real_local() -> None:
    """
    Real Ollama test: load BatchedOllamaEmbedding via load_embedding_model(embedding_type="ollama")
    and get embeddings for multiple texts. Skips if Ollama is not reachable or no embedding model.
    Uses OLLAMA_API_BASE_URL from env (default http://localhost:11434).
    """
    import os
    import urllib.error
    import urllib.request

    base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    except (OSError, urllib.error.URLError) as err:
        pytest.skip(f"Ollama not reachable at {base_url}: {err}")

    model = load_embedding_model(embedding_type="ollama")
    assert isinstance(model, BatchedOllamaEmbedding)

    texts = ["first", "second text", "third"]
    embeddings = model.get_text_embedding_batch(texts)
    assert len(embeddings) == len(texts)
    dim = len(embeddings[0])
    assert dim > 0
    for i, emb in enumerate(embeddings):
        assert len(emb) == dim, f"embedding {i} wrong length"
        assert all(isinstance(x, float) for x in emb), f"embedding {i} not all float"