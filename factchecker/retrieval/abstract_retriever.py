from abc import ABC, abstractmethod
from factchecker.indexing.abstract_indexer import AbstractIndexer

class AbstractRetriever(ABC):
    def __init__(self, indexer: AbstractIndexer, options=None):
        self.indexer = indexer
        self.options = options if options is not None else {}
        # FIXME if we have to pass options to retrieve, we need to separate the options dictionary
        self.top_k = self.options.pop('top_k', 5) # Number of chunks to retrieve
        self.retriever = None

    @abstractmethod
    def create_retriever(self):
        # Ensure the index is created before creating the retriever
        if self.indexer.index is None:
            self.indexer.create_index()
        pass

    @abstractmethod
    def retrieve(self, query):
        # Ensure the retriever is created before retrieving
        if self.retriever is None:
            self.create_retriever()