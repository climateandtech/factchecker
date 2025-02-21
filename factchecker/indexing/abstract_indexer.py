"""Abstract base class for creating and managing indexes."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from llama_index.core import SimpleDirectoryReader, Document

class AbstractIndexer(ABC):
    """
    Abstract base class for creating and managing indexes.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        source_directory (str): Directory containing source data files.
        initial_documents (Optional[list[Document]]): Initial documents to be indexed.
        initial_files (Optional[list[str]]): Initial files to be indexed.
        index (Optional[Any]): In-memory index object.

    """

    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the AbstractIndexer with configuration options.

        Args:
            options (Optional[Dict[str, Any]]): Dictionary containing configuration options. 
                If not provided, defaults will be used.

        """
        self.options: Dict[str, Any] = options if options is not None else {}
        self.index_name: str = self.options.pop('index_name', 'default_index')
        self.index_path: Optional[str] = self.options.pop('index_path', None)
        self.source_directory: str = self.options.pop('source_directory', None)
        self.initial_documents = self.options.pop('documents', None)
        self.initial_files = self.options.pop('files', None)
        self.index: Optional[Any] = None

    def load_initial_documents(self) -> List[Document]:
        """
        Load documents either from preloaded data or from the specified source.

        Returns:
            List[Any]: A list of documents loaded for indexing.

        Raises:
            FileNotFoundError: If the source directory or specified files do not exist.
            Exception: For other issues encountered during document loading.
        """
        try:
            # Use preloaded documents if provided
            if self.initial_documents:
                logging.info("Using preloaded documents provided in options.")
                return self.initial_documents

            # Use specified files if provided
            if self.initial_files:
                files = self.initial_files
                logging.info(f"Loading documents from specified files: {files}")
                documents = SimpleDirectoryReader(input_files=files).load_data()
                self.initial_documents = documents
                logging.debug(f"Loaded {len(documents)} documents from specified files")
                return documents

            # Load from source directory
            if self.source_directory:
                logging.info(f"Loading documents from source directory: {self.source_directory}")
                documents = SimpleDirectoryReader(self.source_directory).load_data()
                self.initial_documents = documents
                logging.debug(f"Loaded {len(documents)} documents from {self.source_directory}")
                return documents
            
            raise ValueError("No documents, files or source directory provided for indexing.")

        except FileNotFoundError as e:
            logging.error(f"File not found during document loading: {e}")
            raise
        except Exception as e:
            logging.exception(f"An error occurred while loading documents: {e}")
            raise

        
    def check_persisted_index_exists(self) -> bool:
        """
        Determine if a persisted index exists at the specified path.

        Returns:
            bool: True if the persisted index exists, False otherwise.
            
        """
        if self.index_path and os.path.exists(self.index_path):
            logging.debug(f"Index exists at {self.index_path}")
            return True
        logging.debug("No persisted index found.")
        return False


    def initialize_index(self) -> None:
        """
        Initializes the index by loading an existing index or building a new one.

        Raises:
            Exception: If an error occurs during index initialization.

        """
        if self.index is not None:
            logging.info("In-memory index already exists. Skipping creation.")
            return
        
        try:
            logging.info("No existing index found. Building a new index...")
            documents = self.load_initial_documents()
            self.build_index(documents)

        except FileNotFoundError as e:
            logging.error(f"File not found during initialization: {e}")
            raise
        except ValueError as e:
            logging.error(f"Invalid data provided: {e}")
            raise

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """
        Builds the index from the provided documents.

        Args:
            documents (List[Any]): The documents to index.

        """
        pass

    @abstractmethod
    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Save the index to persistent storage.
                
        Args:
            index_path (Optional[str]): Path where the index will be saved.
        """
        pass

    @abstractmethod
    def load_index(self) -> None:
        """Loads the index from persistent storage."""
        pass

    @abstractmethod
    def add_to_index(self, documents: List[Any]) -> None:
        """
        Add documents to the index.

        Args:
            documents (List[Any]): Documents to be added to the index.

        """
        pass

    @abstractmethod
    def delete_from_index(self, document_ids: List[Any]) -> None:
        """
        Delete documents from the index based on their IDs.

        Args:
            document_ids (List[Any]): IDs of documents to be removed from the index.

        """
        pass
