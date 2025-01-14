import pytest
from unittest.mock import patch, MagicMock
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Any, List

class MockEmbedding(BaseEmbedding):
    embed_dim: int = 384

    def __init__(self, dim: int = 384, **kwargs: Any):
        super().__init__(model_name="mock", embed_batch_size=10)
        self.embed_dim = dim
        
    def _get_query_embedding(self, query: str) -> List[float]:
        return [0.1] * self.embed_dim
        
    def _get_text_embedding(self, text: str) -> List[float]:
        return [0.1] * self.embed_dim

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return [0.1] * self.embed_dim
        
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return [0.1] * self.embed_dim

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.embed_dim for _ in texts]

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