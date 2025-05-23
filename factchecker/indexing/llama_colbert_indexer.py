"""LlamaIndex ColBERT Indexer """

from typing import Any, Dict, List, Optional
import logging

from llama_index.core import Document
from llama_index.indices.managed.colbert import ColbertIndex

from factchecker.indexing.abstract_indexer import AbstractIndexer

class LlamaColBERTIndexer(AbstractIndexer):
    """ 
    LlamaIndex ColBERT Indexer for creating and managing ColBERT indexes.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        source_directory (str): Directory containing source data files.
        index (Optional[Any]): In-memory index object.
        ---
        gpus (int): number of GPUs to use for indexing.
        show_progress (bool): Whether to show progress during indexing.
    """
    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the LlamaColBERTIndexer with specified parameters.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options which may include:
                - index_name (str): Name of the index. Defaults to 'default_index'.
                - index_path (Optional[str]): Path where the index is stored on disk.
                - source_directory (str): Directory containing source data files. Defaults to 'data'.
                - gpus (int): Number of GPUs to use for indexing.
                - show_progress (bool): Whether to show progress during indexing. 

        """
        super().__init__(options)
        self.gpus = self.options.pop('gpus', 0)
        self.show_progress = self.options.pop('show_progress', False)

    def build_index(self, documents: List[Document]) -> None:
        """
        Builds the LlamaIndex ColBERT index from the provided documents.

        Args:
            documents (List[Document]): List of LlamaIndex Documents to index.

        """
        try:
            # There is currently (28.9.2024) a bug with the LlamaIndex ColBERT Indexing using the from_documents function resulting in:
            # TypeError: ColbertIndex._build_index_from_nodes() got an unexpected keyword argument 'index_name'
            # See: https://github.com/run-llama/llama_index/issues/14398 
            logging.warning("LlamaIndex ColBERT indexing has a known bug. Proceeding may result in errors.")

            self.index = ColbertIndex.from_documents(
                documents,
                gpus=self.gpus,
                show_progress=self.show_progress,
            )

        except Exception as e:
            logging.exception(f"Failed to create LlamaColBERT index: {e}")
            raise

    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Saves the LlamaIndex ColBERT index to disk.

        Args:
            index_path (Optional[str]): The path where the index should be saved.
        """
        logging.error("save_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("save_index() of LlamaColBERTIndexer is not yet implemented")
 
    def load_index(self) -> None:
        """Load the LlamaIndex ColBERT index from disk."""
        logging.error("load_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("load_index() of LlamaColBERTIndexer is not yet implemented")


    def add_to_index(self, documents: List[Document]) -> None:
        """
        Adds documents to the index.

        Args:
            documents (List[Document]): Documents to be added to the index.

        """
        logging.error("add_to_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of LlamaColBERTIndexer is not yet implemented")


    def delete_from_index(self, document_ids: List[str]) -> None:
        """
        Deletes documents from the index.

        Args:
            document_ids (List[str]): List of document IDs to delete from the index.

        """
        logging.error("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
