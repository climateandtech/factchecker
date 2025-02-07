from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from ragatouille import RAGPretrainedModel

class RagatouilleColBERTRetriever(AbstractRetriever):
    def __init__(self, indexer: RagatouilleColBERTIndexer, options=None):
        super().__init__(indexer, options)
        self.options.pop('top_k', None) # Remove top_k from options

    def create_retriever(self):
        super().create_retriever()
        if self.indexer.index is not None:
            if isinstance(self.indexer.index, RAGPretrainedModel):
                # For RAGatouille the retriever is indexing and retrieving is done by the same RAGPretrainedModel instance
                self.retriever = self.indexer.index
            else:
                raise TypeError("The index is not a valid RAGPretrainedModel instance.")
        else:
            raise ValueError("The index is not loaded or does not exist.")

    def retrieve(self, query):
        super().retrieve(query)
        
        return self.retriever.search(
            query, 
            k=self.top_k, 
            **self.options
            )
