import pytest
from unittest.mock import patch
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer

def fake_embedding(text):
    # Return a vector based on a hash or a simple mapping so that identical texts produce identical vectors.
    return [hash(text) % 100]

@pytest.fixture(autouse=True)
def patch_openai_embedding():
    with patch('llama_index.embeddings.openai.OpenAIEmbedding', autospec=False) as mock_embedding_cls:
        instance = mock_embedding_cls.return_value
        instance.embed = fake_embedding  # Patch the 'embed' method
        yield mock_embedding_cls

def test_retrieve_with_llama_base_retriever(get_llama_vector_store_indexer: LlamaVectorStoreIndexer) -> None:
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