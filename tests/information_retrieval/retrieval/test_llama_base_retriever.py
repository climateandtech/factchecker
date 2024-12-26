
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

def test_retrieve_with_llama_base_retriever(get_llama_vector_store_indexer):
    """Test the LlamaBaseRetriever's ability to retrieve documents from the index."""

    indexer = get_llama_vector_store_indexer

    top_k = 2
    retriever_options = {
        'top_k': top_k,
    }
    
    retriever = LlamaBaseRetriever(indexer, retriever_options)
    
    query = "first test document"
    
    # Perform retrieval
    results = retriever.retrieve(query)
    
    assert results is not None
    assert len(results) == top_k
    assert any("first test document" in result.get_text() for result in results), "The results should include the correct document"