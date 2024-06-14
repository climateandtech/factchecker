from factchecker.retrieval.base import BaseRetriever
from ragatouille import RAGPretrainedModel
import os

class ColBERTRetriever(BaseRetriever):
    def __init__(self, indexer, options=None):
        super().__init__(indexer, options)
        self.index_path = self.indexer.index_path
        self.retriever = None

    def create_retriever(self):
        if not os.path.exists(self.index_path):
            print(f"Index not found at {self.index_path}. Creating index...")
            self.indexer.create_index()
            self.index_path = self.indexer.index_path
        else:
            print(f"Index already exists at {self.index_path}")
        
        self.retriever = RAGPretrainedModel.from_index(self.index_path)

    def retrieve(self, query, top_k=5, **kwargs):
        if self.retriever is None:
            self.create_retriever()
        
        results = self.retriever.search(query, k=top_k, **kwargs)
        return results
