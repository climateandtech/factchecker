from factchecker.indexing.base import BaseIndexer

class BaseRetriever:
    def __init__(self, indexer: BaseIndexer, options=None):
        self.indexer = indexer
        self.options = options if options is not None else {}
        # FIXME if we have to pass options to retreive we need to separate it


    def create_retriever(self):
        if not hasattr(self.indexer, 'index'):
            self.indexer.create_index()
        self.retriever = self.indexer.index.as_retriever(**self.options)

    def retrieve(self, query, top_k=5, **kwargs):
        # Ensure the internal retriever is created
        if not hasattr(self, 'retriever'):
            self.create_retriever()
        # Merge options with any additional keyword arguments
        retrieve_options = {**self.options, **kwargs}
        import pdb; pdb.set_trace()
        return self.retriever.retrieve(query)

# Example usage:
# indexer = BaseIndexer()  # Initialize indexer with desired configurations
# base_retriever = BaseRetriever(indexer)
# retriever = base_retriever.create_retriever(similarity_top_k=10, response_mode='tree_summarize')