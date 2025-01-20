from llama_index.core.llms import ChatMessage
from llama_index.core import Document
import pytest
from unittest.mock import Mock, patch
from factchecker.core.llm import load_llm
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from tests.indexing.test_embedding_models import MockEmbedding
import httpx

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