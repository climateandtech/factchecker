import pytest
from unittest.mock import patch, MagicMock
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class MockEmbedding:
    def __init__(self, dim=384):
        self.dim = dim
        
    def _get_query_embedding(self, query: str) -> list:
        return [0.1] * self.dim
        
    def _get_text_embedding(self, text: str) -> list:
        return [0.1] * self.dim

def test_openai_embedding_initialization():
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_EMBEDDING_MODEL': 'text-embedding-ada-002'
    }):
        embedding = OpenAIEmbedding()
        assert embedding is not None
        assert embedding.model_name == 'text-embedding-ada-002'

def test_huggingface_embedding_initialization():
    with patch.dict('os.environ', {
        'OPENAI_EMBEDDING_MODEL': 'BAAI/bge-small-en-v1.5'
    }):
        embedding = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
        assert embedding is not None
        assert 'bge' in embedding.model_name.lower()

@pytest.fixture
def mock_embeddings():
    return MockEmbedding() 