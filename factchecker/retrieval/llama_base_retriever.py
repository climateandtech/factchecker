from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.indexing.abstract_indexer import AbstractIndexer

class LlamaBaseRetriever(AbstractRetriever):
    def __init__(
            self, 
            indexer: AbstractIndexer, 
            options: dict = None
        ) -> None:
        super().__init__(indexer, options)
        self.options.pop('top_k', None) # Remove top_k from options

    def create_retriever(self):
        super().create_retriever()
        self.retriever = self.indexer.index.as_retriever(
            similarity_top_k=self.top_k,
            mode="default",  # This should return NodeWithScore by default
            **self.options
        )

    def retrieve(self, query: str):
        super().retrieve(query)
        return self.retriever.retrieve(query)
