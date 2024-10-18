"""Abstract base class for creating and managing indexes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import os
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)

class AbstractIndexer(ABC):
    """
    Abstract base class for creating and managing indexes.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        source_directory (str): Directory containing source data files.
        index (Optional[Any]): In-memory index object.
    """

    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the AbstractIndexer with specified parameters.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options which may include:
                - index_name (str): Name of the index. Defaults to 'default_index'.
                - index_path (Optional[str]): Path to the directory where the index is stored on disk.
                - source_directory (str): Directory containing source data files. Defaults to 'data'.

        """
        self.options: Dict[str, Any] = options if options is not None else {}
        self.index_name: str = self.options.pop('index_name', 'default_index')
        self.index_path: Optional[str] = self.options.pop('index_path', None)
        self.source_directory: str = self.options.pop('source_directory', 'data')
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
            documents = self.options.get('documents')
            if documents:
                logger.info("Using preloaded documents provided in options.")
                return documents

            # Use specified files if provided
            files = self.options.get('files')
            if files:
                logger.info(f"Loading documents from specified files: {files}")
                documents = SimpleDirectoryReader(input_files=files).load_data()
                logger.debug(f"Loaded {len(documents)} documents from specified files")
                return documents

            # Load from source directory as default
            logger.info(f"Loading documents from source directory: {self.source_directory}")
            documents = SimpleDirectoryReader(self.source_directory).load_data()
            logger.debug(f"Loaded {len(documents)} documents from {self.source_directory}")
            return documents

        except FileNotFoundError as e:
            logger.error(f"File not found during document loading: {e}")
            raise
        except Exception as e:
            logger.exception(f"An error occurred while loading documents: {e}")
            raise

        
    def check_persisted_index_exists(self) -> bool:
        """
        Determine if a persisted index exists at the specified path.

        Returns:
            bool: True if the persisted index exists, False otherwise.
        """
        if self.index_path and os.path.exists(self.index_path):
            logger.debug(f"Index exists at {self.index_path}")
            return True
        logger.debug("No persisted index found.")
        return False


    def initialize_index(self) -> None:
        """
        Initializes the index by loading an existing index or building a new one.

        Raises:
            Exception: If an error occurs during index initialization.
        """

        try:
            if self.index is not None:
                logger.info("In-memory index already exists. Skipping creation.")
                return

            if self.check_persisted_index_exists():
                logger.info(f"Persisted index found at {self.index_path}. Loading index...")
                self.load_index()
                return

            logger.info("No existing index found. Building a new index...")
            documents = self.load_initial_documents()
            self.build_index(documents)

        except Exception as e:
            logger.exception(f"An error occurred during index creation: {e}")
            raise

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """
        Builds the index from the provided documents.

        Args:
            documents (List[Any]): The documents to index.

        """
        pass

    def save_index(self, index_path: Optional[str] = None) -> None:
        """Save the index to persistent storage."""
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
