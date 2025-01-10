import sys 
import os 
import pytest

# Add the factchecker module to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["IS_TESTING"] = "true"
    # Configure mock embeddings for testing
    os.environ["LLAMA_INDEX_EMBED_MODEL"] = "local"
    os.environ["MOCK_EMBED_DIM"] = "8"