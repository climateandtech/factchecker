
from factchecker.indexing.abstract_indexer import AbstractIndexer
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
import logging

logger = logging.getLogger(__name__)

class LlamaVectorStoreIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.embedding_model = self.options.pop('embedding_model', None)
        self.vector_store = self.options.pop('vector_store', None)
        self.transformations = self.options.pop('transformations', [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)])
        self.show_progress = self.options.pop('show_progress', False)

    def check_persisted_index_exists(self):
        # TODO: Implement the method to check if the index exists on disk
        pass

    def create_index(self):

        # Stop further execution if the index already exists or was loaded
        if super().create_index():
            return  

        try:

            # Now self.options should only contain relevant options for StorageContext.from_defaults
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store, **self.options)
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                transformations=self.transformations,
            )
        
        except Exception as e:
            logger.exception(f"Failed to create LlamaVectorStore index: {e}")
            raise

    def load_index(self):
        logger.error("load_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("load_index of LlamaVectorStoreIndexer is not yet implemented")

    def add_to_index(self, documents):
        logger.error("add_to_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("add_to_index of LlamaVectorStoreIndexer is not yet implemented")

    def delete_from_index(self, document_ids):
        logger.error("delete_from_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index of LlamaVectorStoreIndexer is not yet implemented")
