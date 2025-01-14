import pytest
from unittest.mock import patch
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

@pytest.fixture
def mock_embeddings():
    """Mock embeddings to avoid OpenAI calls"""
    with patch('llama_index.embeddings.openai.OpenAIEmbedding') as mock:
        yield mock

def test_retrieve_with_llama_base_retriever(get_llama_vector_store_indexer, mock_embeddings):
    """Test the LlamaBaseRetriever's ability to retrieve documents from the index."""

    indexer = get_llama_vector_store_indexer

    top_k = 2
    retriever_options = {
        'top_k': top_k,  # Retrieve top 2 documents
    }
    
    retriever = LlamaBaseRetriever(indexer, retriever_options)
    
    query = "This is the first test document."  # Exact match with test document
    
    # Perform retrieval
    results = retriever.retrieve(query)
    
    # Assertions
    assert results is not None, "Results should not be None"
    assert len(results) == top_k, "Two documents should be retrieved"
    assert any("This is the first test document" in result.get_text() for result in results), "The results should include the correct document"