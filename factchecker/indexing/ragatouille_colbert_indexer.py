"""Ragatouille ColBERT Indexer."""

import os
import logging
from typing import Any, Dict, List, Optional

from ragatouille import RAGPretrainedModel
from llama_index.core import Document

from factchecker.indexing.abstract_indexer import AbstractIndexer


class RagatouilleColBERTIndexer(AbstractIndexer):
    """
    RagatouilleColBERTIndexer for creating and managing ColBERT indexes using Ragatouille.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        index (Optional[Any]): In-memory index object.
        ---
        max_document_length (int): Maximum length of documents during indexing.
        checkpoint (str): Pretrained model checkpoint to use.
        overwrite_index (bool): Whether to overwrite an existing index.
        index_root (str): Root directory where indexes are stored.

    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the RagatouilleColBERTIndexer with specified parameters.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options which may include:
                - index_name (str): Name of the index. Defaults to 'default_index'.
                - index_path (Optional[str]): Path to the directory where the index is stored on disk.
                - source_directory (str): Directory containing source data files. Defaults to 'data'.
                - max_document_length (int): Maximum length of documents during indexing.
                - checkpoint (str): Pretrained model checkpoint to use.
                - overwrite_index (bool): Whether to overwrite an existing index.
                - index_root (str): Root directory where indexes are stored.

        """
        super().__init__(options)
        self.max_document_length: int = self.options.pop('max_document_length', 256)
        self.checkpoint: str = self.options.pop('checkpoint', 'colbert-ir/colbertv2.0')
        self.overwrite_index: bool = self.options.pop('overwrite_index', True)
        self.index_root: str = self.options.pop('index_root', 'indexes/ragatouille/')
        if not os.path.isdir(self.index_root):
            logging.warning(f"Index root {self.index_root} does not exist. Creating it.")
            os.makedirs(self.index_root, exist_ok=True)


    def build_index(self, documents: List[Document]) -> None:
        """ 
        Builds the Ragatouille ColBERT index from the provided documents.

        Side effect: The index is stored on disk at self.index_path.

        Args:
            documents (List[Document]): List of LlamaIndex Documents to index.

        Raises:
            Exception: If an error occurs during index creation.
        """
        try:
            self.index = RAGPretrainedModel.from_pretrained(
                self.checkpoint,
                index_root=self.index_root, 
            )

            # RAGatouille requires a list of texts to create the index
            texts = [document.text for document in documents if hasattr(document, 'text')]

            # RAGatouille automatically saves the created index to disc
            self.index_path = self.index.index(
                collection=texts,
                index_name=self.index_name,
                max_document_length=self.max_document_length,
                overwrite_index=self.overwrite_index,
            )
            
            logging.info(f"Index created and stored at {self.index_path}")

        except Exception as e:
            logging.exception(f"Failed to create RagatouilleColBERT index: {e}")
            raise

    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Saves the Ragatouille ColBERT index to disk.

        Args:
            index_path (Optional[str]): The path where the index should be saved.
        """
        logging.error("save_index() of RagatouilleColBERTIndexer is not implemented because indexing saves automatically.")
        raise NotImplementedError("save_index() of RagatouilleColBERTIndexer is not implemented")

    def load_index(self) -> None:
        """ 
        Loads the Ragatouille ColBERT index from disk into memory.

        Raises:
            Exception: If an error occurs during index loading.
        """
        try:
            if self.check_persisted_index_exists():
                self.index = RAGPretrainedModel.from_index(self.index_path)
                logging.info(f"Index loaded from {self.index_path}")
            else:
                logging.error(f"No index found at {self.index_path}")
                raise FileNotFoundError(f"No index found at {self.index_path}")
        except Exception as e:
            logging.exception(f"Failed to load index from {self.index_path}: {e}")
            raise

    def add_to_index(self, documents: List[Document]) -> None:
        """
        Adds documents to the existing index.

        Args:
            documents (List[Document]): Documents to be added to the index.
        """
        logging.error("add_to_index() of RagatouilleColBERTIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of RagatouilleColBERTIndexer is not yet implemented")

    def delete_from_index(self, document_ids: List[str]) -> None:
        """
        Deletes documents from the index.

        Args:
            document_ids (List[str]): List of document IDs to delete from the index.
        """
        logging.error("delete_from_index() of RagatouilleColBERTIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of RagatouilleColBERTIndexer is not yet implemented")

