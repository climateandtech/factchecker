from factchecker.indexing.abstract_indexer import AbstractIndexer
from llama_index.indices.managed.colbert import ColbertIndex
import logging

logger = logging.getLogger(__name__)

class LlamaColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.gpus = self.options.pop('gpus', '0')
        self.show_progress = self.options.pop('show_progress', False)

    def check_persisted_index_exists(self):
        # TODO: Implement the method to check if the index exists on disk
        pass

    def build_index(self, documents):
        try:
            # There is currently (28.9.2024) a bug with the LlamaIndex ColBERT Indexing using the from_documents function resulting in:
            # TypeError: ColbertIndex._build_index_from_nodes() got an unexpected keyword argument 'index_name'
            # See: https://github.com/run-llama/llama_index/issues/14398 
            logger.warning("LlamaIndex ColBERT indexing has a known bug. Proceeding may result in errors.")

            self.index = ColbertIndex.from_documents(
                documents,
                gpus=self.gpus,
                show_progress=self.show_progress,
            )

        except Exception as e:
            logger.exception(f"Failed to create LlamaColBERT index: {e}")
            raise

    def save_index(self):
        logger.error("save_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("save_index() of LlamaColBERTIndexer is not yet implemented")
 
    def load_index(self):
        logger.error("load_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("load_index() of LlamaColBERTIndexer is not yet implemented")


    def add_to_index(self, documents):
        logger.error("add_to_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of LlamaColBERTIndexer is not yet implemented")


    def delete_from_index(self, document_ids):
        logger.error("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
