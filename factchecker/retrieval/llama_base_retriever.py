from factchecker.retrieval.abstract_retriever import AbstractRetriever

class LlamaBaseRetriever(AbstractRetriever):
    def __init__(self, indexer, options=None):
        super().__init__(indexer, options)

    def create_retriever(self):
        super().create_retriever()
        retriever_options = self.options.copy()
        retriever_options.pop('similarity_top_k', None)
        
        self.retriever = self.indexer.index.as_retriever(
            similarity_top_k=self.top_k,
            **retriever_options
        )

    def retrieve(self, query):
        super().retrieve(query)
        return self.retriever.retrieve(query)
