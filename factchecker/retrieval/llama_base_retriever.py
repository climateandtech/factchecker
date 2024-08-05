from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer

class LlamaBaseRetriever(AbstractRetriever):
    def __init__(self, indexer, options=None):
        super().__init__(indexer, options)

    def create_retriever(self):
        self.retriever = self.indexer.index.as_retriever(**self.options)

    def retrieve(self, query):
        super().retrieve(query)
        return self.retriever.retrieve(query)

# Example usage:
# TODO: Add example usage