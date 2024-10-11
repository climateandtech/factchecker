"""LlamaIndex ColBERT Indexer """

from factchecker.indexing.abstract_indexer import AbstractIndexer
from llama_index.indices.managed.colbert import ColbertIndex
from llama_index.core import Document
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LlamaColBERTIndexer(AbstractIndexer):
    """ 
    LlamaIndex ColBERT Indexer

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        source_directory (str): Directory containing source data files.
        index (Optional[Any]): In-memory index object.
        gpus (int): number of GPUs to use for indexing.
        show_progress (bool): Whether to show progress during indexing.
    """
    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LlamaColBERTIndexer with specified parameters.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options which may include:
                - index_name (str): Name of the index. Defaults to 'default_index'.
                - index_path (Optional[str]): Path to the directory where the index is stored on disk.
                - source_directory (str): Directory containing source data files. Defaults to 'data'.
                - gpus (int): number of GPUs to use for indexing. Defaults to 0.
                - show_progress (bool): Whether to show progress during indexing. Defaults
        """
        super().__init__(options)
        self.gpus = self.options.pop('gpus', 0)
        self.show_progress = self.options.pop('show_progress', False)

    def build_index(self, documents: list[Document]) -> None:
        """
        Build the LlamaIndex ColBERT index from the provided documents.

        Args:
            documents (list[Document]): List of LlamaIndex Documents to index.

        """
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

    def save_index(self) -> None:
        """Save the LlamaIndex ColBERT index to disk."""
        logger.error("save_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("save_index() of LlamaColBERTIndexer is not yet implemented")
 
    def load_index(self) -> None:
        """Load the LlamaIndex ColBERT index from disk."""
        logger.error("load_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("load_index() of LlamaColBERTIndexer is not yet implemented")


    def add_to_index(self, documents) -> None:
        """Add documents to the index."""
        logger.error("add_to_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of LlamaColBERTIndexer is not yet implemented")


    def delete_from_index(self, document_ids) -> None:
        """
        Delete documents from the index.

        Args:
            document_ids (List[str]): List of document IDs to delete from the index 
        """
        logger.error("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
