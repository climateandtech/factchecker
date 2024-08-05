
from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from ragatouille import RAGPretrainedModel
import os

class RagatouilleColBERTRetriever(AbstractRetriever):
    def __init__(self, indexer: RagatouilleColBERTIndexer, options=None):
        super().__init__(indexer, options)

    def create_retriever(self):
        if not self.indexer.check_index_exists():
            print(f"Index not found at {self.indexer.index_path}. Creating index...")
            self.indexer.create_index()
        else:
            print(f"Index found at {self.indexer.index_path}. Using existing index for retrieval.")
        
        self.retriever = RAGPretrainedModel.from_index(self.indexer.index_path, **self.options)

    def retrieve(self, query):
        super().retrieve(query)
        return self.retriever.search(query, k=self.top_k)