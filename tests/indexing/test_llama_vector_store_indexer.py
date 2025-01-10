from unittest.mock import patch, MagicMock
import pytest
from llama_index.core.embeddings import BaseEmbedding

# ... existing imports ...

class MockEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]:
        # Return a fixed-size mock embedding vector
        return [0.1] * 384  # or whatever dimension you want to use

    def _get_text_embedding(self, text: str) -> List[float]:
        # Return a fixed-size mock embedding vector
        return [0.1] * 384

@pytest.fixture
def mock_embeddings():
    with patch('llama_index.core.embeddings.OpenAIEmbedding', MockEmbedding):
        yield

@pytest.mark.usefixtures("mock_embeddings")
class TestLlamaVectorStoreIndexer:
    # ... existing test methods ...