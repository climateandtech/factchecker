from abc import ABC, abstractmethod
from factchecker.indexing.abstract_indexer import AbstractIndexer

class AbstractRetriever(ABC):
    def __init__(self, indexer: AbstractIndexer, options=None):
        self.indexer = indexer

        self.options = options if options is not None else {}
        self.top_k = self.options.pop('top_k', 5)
            # FIXME if we have to pass options to retrieve, we need to separate it
        # Ensure the index is created during initialization
        self.create_index()
        self.retriever = None

    def create_index(self):
        if self.indexer.check_index_exists() is False:
            self.indexer.create_index()

    @abstractmethod
    def create_retriever(self):
        pass

    @abstractmethod
    def retrieve(self, query):
        # Ensure the internal retriever is created
        if self.retriever is None:
            self.create_retriever()