
from .abstract_indexer import AbstractIndexer
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

class LlamaVectorStoreIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.embedding_model = self.options.pop('embedding_model', None)
        self.vector_store = self.options.pop('vector_store', None)
        self.transformations = self.options.pop('transformations', [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)])
        self.show_progress = self.options.pop('show_progress', False)

    def check_persisted_index_exists(self):
        # TODO: first implement the methods to save and load this type of index
        # TODO: then implement this method to check if the index exists on disk
        pass

    def create_index(self):

        if super().create_index():
            return  # Stop further execution if the index already exists or was loaded

        # Now self.options should only contain relevant options for StorageContext.from_defaults
        # TODO: check if StorageContext should really take all the remaining options as it will block them from being passed to the VectorStoreIndex.from_documents method which may need to receive additional parameters - The parameters extracted from options in the init function are not exhaustive for VectorStoreIndex, see: https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store, **self.options)
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            embed_model=self.embedding_model,
            transformations=self.transformations,
            # TODO: check if any additional parameters should be passed here
        )

    def load_index(self):
        # TODO: implement
        pass

    def add_to_index(self, documents):
        # TODO: implement
        pass

    def delete_from_index(self, document_ids):
        # TODO: implement
        pass
