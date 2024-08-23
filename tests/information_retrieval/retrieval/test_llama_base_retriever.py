import pytest
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

def test_retrieve_with_llama_base_retriever(prepare_llama_vector_store_indexer):
    """Test the LlamaBaseRetriever's ability to retrieve documents from the index."""

    top_k = 2
    
    # Setup retriever with the indexer
    retriever_options = {
        'top_k': top_k,  # Retrieve top 2 documents
    }
    
    retriever = LlamaBaseRetriever(prepare_llama_vector_store_indexer, retriever_options)
    
    query = "first test document"
    
    # Perform retrieval
    results = retriever.retrieve(query)
    
    # Assertions
    assert results is not None, "Results should not be None"
    assert len(results) == top_k, "Two documents should be retrieved"
    assert any("first test document" in result.get_text() for result in results), "The results should include the correct document"