from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from unittest.mock import patch
import numpy as np

@patch('llama_index.embeddings.openai.OpenAIEmbedding')
def test_retrieve_with_llama_base_retriever(mock_embedding, get_llama_vector_store_indexer):
    """Test the LlamaBaseRetriever's ability to retrieve documents from the index."""

    # Configure mock embeddings
    mock_embedding.return_value._get_text_embedding.return_value = np.array([0.1] * 384)
    mock_embedding.return_value._get_query_embedding.return_value = np.array([0.1] * 384)
    
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