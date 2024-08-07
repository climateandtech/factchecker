from abc import ABC, abstractmethod
from factchecker.indexing.abstract_indexer import AbstractIndexer

class AbstractRetriever(ABC):
    def __init__(self, indexer: AbstractIndexer, options=None):
        self.indexer = indexer
        self.options = options if options is not None else {}
        # FIXME if we have to pass options to retrieve, we need to separate the options dictionary
        self.top_k = self.options.pop('top_k', 5) # Number of chunks to retrieve
        self.create_index() # Ensure the index is created during initialization
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