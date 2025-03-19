import pytest
from unittest.mock import patch, MagicMock
from factchecker.core.llm import load_llm
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Document
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
import httpx
from llama_index.core.llms import ChatMessage
from unittest.mock import Mock

@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to set up a clean environment for each test"""
    env_vars = {
        "LLM_TYPE": None,
        "OPENAI_API_MODEL": None,
        "OPENAI_API_KEY": None,
        "OPENAI_ORGANIZATION": None,
        "OPENAI_API_BASE": None,
        "TEMPERATURE": None,
        "OLLAMA_MODEL": None,
        "OLLAMA_REQUEST_TIMEOUT": None,
        "OLLAMA_API_BASE_URL": None
    }
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch

def test_load_llm_default_openai(mock_env):
    """Test loading OpenAI LLM with default settings"""
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    
    llm = load_llm()
    
    assert isinstance(llm, OpenAI)
    assert llm.model == "gpt-3.5-turbo-1106"
    assert llm.temperature == 0.1  # Default from env
    assert llm.api_key == "test-key"

def test_load_llm_zero_temperature(mock_env):
    """Test that explicitly setting temperature=0.0 is respected"""
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    mock_env.setenv("TEMPERATURE", "0.5")  # Should be ignored
    
    llm = load_llm(temperature=0.0)
    
    assert isinstance(llm, OpenAI)
    assert llm.temperature == 0.0  # Should keep explicit 0.0

def test_load_llm_custom_openai_params(mock_env):
    """Test loading OpenAI LLM with custom parameters"""
    llm = load_llm(
        model="gpt-4",
        temperature=0.7,
        api_key="custom-key",
        organization="org-123",
        api_base="custom-base"
    )
    
    assert isinstance(llm, OpenAI)
    assert llm.model == "gpt-4"
    assert llm.temperature == 0.7
    assert llm.api_key == "custom-key"
    assert llm.api_base == "custom-base"

def test_load_llm_from_env(mock_env):
    """Test loading LLM with parameters from environment variables"""
    mock_env.setenv("OPENAI_API_MODEL", "gpt-4")
    mock_env.setenv("OPENAI_API_KEY", "env-key")
    mock_env.setenv("OPENAI_ORGANIZATION", "env-org")
    mock_env.setenv("OPENAI_API_BASE", "env-base")
    mock_env.setenv("TEMPERATURE", "0.5")
    
    llm = load_llm()
    
    assert isinstance(llm, OpenAI)
    assert llm.model == "gpt-4"
    assert llm.temperature == 0.5  # From env
    assert llm.api_key == "env-key"
    assert llm.api_base == "env-base"

def test_load_llm_ollama(mock_env):
    """Test loading Ollama LLM"""
    mock_env.setenv("LLM_TYPE", "ollama")
    mock_env.setenv("OLLAMA_MODEL", "llama2")
    mock_env.setenv("OLLAMA_REQUEST_TIMEOUT", "60.0")
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    llm = load_llm()
    
    assert isinstance(llm, Ollama)
    assert llm.model == "llama2"
    assert llm.request_timeout == 60.0
    assert llm.base_url == "http://localhost:11434"

def test_load_llm_ollama_custom_params(mock_env):
    """Test loading Ollama LLM with custom parameters"""
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")  # Required base URL
    
    llm = load_llm(
        llm_type="ollama",
        model="mistral",
        temperature=0.8,
        request_timeout=30.0
    )
    
    assert isinstance(llm, Ollama)
    assert llm.model == "mistral"
    assert llm.temperature == 0.8
    assert llm.request_timeout == 30.0

def test_load_llm_ollama_with_context_window(mock_env):
    """Test loading Ollama LLM with custom context window"""
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    llm = load_llm(
        llm_type="ollama",
        model="llama2",
        context_window=4000
    )
    
    assert isinstance(llm, Ollama)
    assert llm.context_window == 4000

def test_load_llm_ollama_filters_retriever_kwargs(mock_env):
    """Test that retriever-specific kwargs are filtered out for Ollama"""
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    
    llm = load_llm(
        llm_type="ollama",
        model="llama2",
        top_k=5,  # Should be filtered out
        similarity_top_k=3  # Should be filtered out
    )
    
    assert isinstance(llm, Ollama)
    assert not hasattr(llm, "top_k")
    assert not hasattr(llm, "similarity_top_k")

def test_load_llm_openai_filters_retriever_kwargs(mock_env):
    """Test that retriever-specific kwargs are filtered out for OpenAI"""
    llm = load_llm(
        api_key="test-key",
        top_k=5,  # Should be filtered out
        similarity_top_k=3  # Should be filtered out
    )
    
    assert isinstance(llm, OpenAI)
    assert not hasattr(llm, "top_k")
    assert not hasattr(llm, "similarity_top_k")

def test_load_llm_unknown_type_fallback(mock_env):
    """Test that unknown LLM type falls back to OpenAI"""
    mock_env.setenv("LLM_TYPE", "unknown_llm")
    mock_env.setenv("OPENAI_API_KEY", "test-key")
    
    llm = load_llm()
    
    assert isinstance(llm, OpenAI)
    assert llm.model == "gpt-3.5-turbo-1106"  # Default OpenAI model
    assert llm.api_key == "test-key" 

def test_load_llm_ollama_zero_temperature(mock_env):
    """Test that explicitly setting temperature=0.0 is respected in Ollama"""
    mock_env.setenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    mock_env.setenv("TEMPERATURE", "0.5")  # Should be ignored
    
    llm = load_llm(
        llm_type="ollama",
        model="llama2",
        temperature=0.0
    )
    
    assert isinstance(llm, Ollama)
    assert llm.temperature == 0.0  # Should keep explicit 0.0


@pytest.mark.integration
def test_ollama_integration(mock_embedding):
    # Mock only Ollama-specific environment variables
    with patch.dict('os.environ', {
        'LLM_TYPE': 'ollama',
        'OLLAMA_API_BASE_URL': 'http://localhost:11434',
        'OLLAMA_MODEL': 'llama2'
    }):
        # Create mock embedding model
        mock_embed_model = mock_embedding
        
        # Create test documents
        documents = [
            Document(text="Test document 1"),
            Document(text="Test document 2")
        ]
        
        # Mock Ollama response
        mock_response = ChatMessage(role="assistant", content="Test response")
        
        # Mock the Ollama client's HTTP requests
        mock_http_response = Mock(spec=httpx.Response)
        mock_http_response.json.return_value = {"message": {"role": "assistant", "content": "Test response"}}
        mock_http_response.status_code = 200
        
        with patch('httpx.Client.request', return_value=mock_http_response):
            # Load the LLM
            llm = load_llm()
            
            # Create indexer with mock embedding
            indexer_options = {
                'documents': documents,
                'index_name': 'test_ollama_integration',
                'embedding_kwargs': {'embed_model': mock_embed_model}
            }
            indexer = LlamaVectorStoreIndexer(indexer_options)
            indexer.initialize_index()
            
            # Create retriever
            retriever_options = {'top_k': 2}
            retriever = LlamaBaseRetriever(indexer, retriever_options)
            
            # Test query
            query = "Test query"
            nodes = retriever.retrieve(query)
            
            # Verify the response
            assert nodes is not None
            assert len(nodes) > 0 