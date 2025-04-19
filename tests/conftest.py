import sys 
import os 
import pytest
from unittest.mock import patch
from typing import List

# Add the factchecker module to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockEmbedding:
    """Mock embedding class for testing."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _get_mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding based on text content."""
        # Use a simple hash of the text to generate a deterministic embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to floats between -1 and 1
        embedding = []
        for i in range(self.dim):
            byte_val = hash_bytes[i % len(hash_bytes)]
            embedding.append((byte_val / 127.5) - 1.0)
        return embedding

    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_mock_embedding(text)

    async def aget_text_embedding(self, text: str) -> List[float]:
        return self._get_mock_embedding(text)

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_mock_embedding(text) for text in texts]

    async def aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_mock_embedding(text) for text in texts]

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        return [self._get_mock_embedding(text) for text in texts]

    def get_agg_embedding_from_queries(self, queries: List[str]) -> List[float]:
        """Return an aggregate embedding for a list of queries."""
        # Get embeddings for all queries
        embeddings = [self._get_mock_embedding(query) for query in queries]
        
        # Average the embeddings
        agg_embedding = [0.0] * self.dim
        for emb in embeddings:
            for i in range(self.dim):
                agg_embedding[i] += emb[i]
        
        for i in range(self.dim):
            agg_embedding[i] /= len(queries)
            
        return agg_embedding

@pytest.fixture(autouse=True)
def mock_openai_env(monkeypatch):
    """Mock OpenAI-related environment variables and classes for all tests."""
    monkeypatch.setenv("EMBEDDING_TYPE", "mock")
    monkeypatch.setenv("MOCK_EMBED_DIM", "384")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TYPE", "mock")

    # Mock OpenAI class
    class MockOpenAI:
        def __init__(self, model=None, temperature=None, api_key=None, organization=None, api_base=None, **kwargs):
            self.model = model or "gpt-3.5-turbo-1106"
            self.temperature = temperature or 0.1
            self.api_key = api_key
            self.organization = organization
            self.api_base = api_base
            self.kwargs = kwargs

        def chat(self, messages, **kwargs):
            return {"message": {"role": "assistant", "content": "Test response"}}

    # Mock OpenAIEmbedding class
    class MockOpenAIEmbedding:
        def __init__(self, model_name=None, api_key=None, api_base=None, **kwargs):
            self.model_name = model_name or "text-embedding-ada-002"
            self.api_key = api_key
            self.api_base = api_base
            self.kwargs = kwargs

        def get_text_embedding(self, text):
            return [0.0] * 384

    # Apply mocks
    monkeypatch.setattr("llama_index.llms.openai.OpenAI", MockOpenAI)
    monkeypatch.setattr("llama_index.embeddings.openai.OpenAIEmbedding", MockOpenAIEmbedding)

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["IS_TESTING"] = "true"
    # Configure mock embeddings for testing
    os.environ["LLAMA_INDEX_EMBED_MODEL"] = "local"
    os.environ["MOCK_EMBED_DIM"] = "8"

@pytest.fixture
def get_test_data_directory(tmp_path):
    """Creates a temporary directory with dummy text files for indexing tests."""
    # Create a temporary directory to act as the data source
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create dummy text files in the directory
    for i in range(5):
        with open(data_dir / f"test_file_{i}.txt", 'w') as f:
            f.write(f"This is the content of test file {i}.\n" * 10)
    
    # Return the path to the test directory
    return str(data_dir)