from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, Settings, Document
import pytest
from unittest.mock import Mock, patch
from factchecker.core.llm import load_llm
from tests.indexing.test_embedding_models import MockEmbedding
import httpx

def create_query_engine(documents, embed_model, llm):
    # Configure settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Return query engine
    return index.as_query_engine()

def test_ollama_integration():
    # Mock the environment variables
    with patch.dict('os.environ', {
        'LLM_TYPE': 'ollama',
        'OLLAMA_API_BASE_URL': 'http://localhost:11434',
        'OLLAMA_MODEL': 'llama2'
    }):
        # Create mock embedding model
        mock_embed_model = MockEmbedding(dim=384)
        
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
            
            # Create query engine with mock embedding
            query_engine = create_query_engine(documents, mock_embed_model, llm)
            
            # Test query
            query = "Test query"
            response = query_engine.query(query)
            
            # Verify the response
            assert response is not None
            assert str(response) == "Test response" 