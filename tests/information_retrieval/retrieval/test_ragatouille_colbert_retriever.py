
import pytest

from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from factchecker.retrieval.ragatouille_colbert_retriever import RagatouilleColBERTRetriever


# @pytest.mark.integration
# def test_retrieve_with_ragatouille_colbert_retriever(
#         get_ragatouille_colbert_indexer: RagatouilleColBERTIndexer
#     ) -> None:
#     """Test the RagatouilleColBERTRetriever's ability to retrieve documents from the index."""
#     indexer = get_ragatouille_colbert_indexer

#     top_k = 2
    
#     # Setup retriever with the indexer
#     retriever_options = {
#         'top_k': top_k,  # Retrieve top 2 documents
#     }
    
#     retriever = RagatouilleColBERTRetriever(indexer, retriever_options)
    
#     query = "first test document"
    
#     # Perform retrieval
#     results = retriever.retrieve(query)
    
#     # Assertions
#     assert results is not None
#     assert len(results) == top_k
#     assert any("first test document" in result["content"] for result in results), "The results should include the correct document"