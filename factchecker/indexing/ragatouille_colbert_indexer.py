import os
from factchecker.indexing.abstract_indexer import AbstractIndexer
from ragatouille import RAGPretrainedModel
import logging

logger = logging.getLogger(__name__)

class RagatouilleColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.max_document_length = self.options.pop('max_document_length', 256)
        self.checkpoint = self.options.pop('checkpoint', 'colbert-ir/colbertv2.0')
        self.overwrite_index = self.options.pop('overwrite_index', True)
        self.index_root = self.options.pop('index_root', 'indexes/ragatouille/') # The root directory where indexes will be stored. If None, will use the default directory, '.ragatouille/'.

    def check_persisted_index_exists(self) -> bool:
        return self.index_path and os.path.exists(self.index_path)

    def build_index(self):
        try:
            self.index = RAGPretrainedModel.from_pretrained(
                self.checkpoint,
                index_root=self.index_root, 
            )

            # RAGatouille requires a list of texts to create the index
            texts = [document.text for document in self.documents if hasattr(document, 'text')]

            # RAGatouille automatically saves the created index to disc
            self.index_path = self.index.index(
                collection=texts,
                index_name=self.index_name,
                max_document_length=self.max_document_length,
            )
            
            logger.info(f"Index created and stored at {self.index_path}")

        except Exception as e:
            logger.exception(f"Failed to create RagatouilleColBERT index: {e}")
            raise

    def save_index(self):
        logger.error("save_index() of RagatouilleColBERTIndexer is not yet implemented")
        raise NotImplementedError("save_index() of RagatouilleColBERTIndexer is not yet implemented")

    def load_index(self):
        try:
            if self.check_persisted_index_exists():
                self.index = RAGPretrainedModel.from_index(self.index_path)
                logger.info(f"Index loaded from {self.index_path}")
            else:
                logger.info(f"No index found at {self.index_path}")
        except Exception as e:
            logger.exception(f"Failed to load index from {self.index_path}: {e}")
            raise

    def add_to_index(self, documents):
        logger.error("add_to_index() of RagatouilleColBERTIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of RagatouilleColBERTIndexer is not yet implemented")

    def delete_from_index(self, document_ids):
        logger.error("delete_from_index() of RagatouilleColBERTIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of RagatouilleColBERTIndexer is not yet implemented")

