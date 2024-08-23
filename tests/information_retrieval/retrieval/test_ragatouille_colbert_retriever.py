import pytest
from factchecker.retrieval.ragatouille_colbert_retriever import RagatouilleColBERTRetriever

def test_retrieve_with_ragatouille_colbert_retriever(prepare_ragatouille_colbert_indexer):
    """Test the LlamaBaseRetriever's ability to retrieve documents from the index."""

    indexer = prepare_ragatouille_colbert_indexer

    top_k = 2
    
    # Setup retriever with the indexer
    retriever_options = {
        'top_k': top_k,  # Retrieve top 2 documents
    }
    
    retriever = RagatouilleColBERTRetriever(indexer, retriever_options)
    
    query = "first test document"
    
    # Perform retrieval
    results = retriever.retrieve(query)
    
    # Assertions
    assert results is not None, "Results should not be None"
    assert len(results) == top_k, "Two documents should be retrieved"
    assert any("first test document" in result["content"] for result in results), "The results should include the correct document"