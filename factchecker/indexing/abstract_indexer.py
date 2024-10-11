from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)

class AbstractIndexer(ABC):
    """
    Abstract base class for creating and managing indexes.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        source_directory (str): Directory containing source data files.
        files (Optional[List[str]]): Specific files to include in the index.
        documents (Optional[List[Any]]): Preloaded LlamaIndex documents for indexing.
        index (Optional[Any]): In-memory index object.
    """
    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AbstractIndexer with specified parameters.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options which may include:
                - index_name (str): Name of the index. Defaults to 'default_index'.
                - index_path (Optional[str]): Path to the directory where the index is stored on disk.
                - source_directory (str): Directory containing source data files. Defaults to 'data'.
                - files (Optional[List[str]]): Specific files to include in the index.
                - documents (Optional[List[Any]]): Preloaded documents for indexing.
        """
        self.options = options if options is not None else {}
        self.index_name = self.options.pop('index_name', 'default_index')
        self.index_path = self.options.pop('index_path', None)
        self.source_directory = self.options.pop('source_directory', 'data')
        self.files = self.options.pop('files', None)
        self.documents = self.options.pop('documents', None)
        self.index = None 

    def load_documents(self) -> List[Any]:
        """
        Load documents either from preloaded data or from the specified source.

        Returns:
            List[Any]: A list of documents loaded for indexing.

        Raises:
            FileNotFoundError: If the source directory or specified files do not exist.
            Exception: For other issues encountered during document loading.
        """
        try:
            if self.documents:
                logger.info("Using preloaded documents.")
                return self.documents
            if not self.files:
                logger.info(f"Loading documents from source directory: {self.source_directory}")
                documents = SimpleDirectoryReader(self.source_directory).load_data()
                logger.debug(f"Loaded {len(documents)} documents from {self.source_directory}")
                return documents
            else:
                logger.info(f"Loading documents from specified files: {self.files}")
                documents = SimpleDirectoryReader(input_files=self.files).load_data()
                logger.debug(f"Loaded {len(documents)} documents from specified files")
                return documents
        except FileNotFoundError as e:
            logger.error(f"File not found during document loading: {e}")
            raise
        except Exception as e:
            logger.exception(f"An error occurred while loading documents: {e}")
            raise
        
    @abstractmethod
    def check_persisted_index_exists(self) -> bool:
        """
        Determine if a persisted index exists at the specified path.

        Returns:
            bool: True if the persisted index exists, False otherwise.
        """
        pass

    @abstractmethod
    def create_index(self):
        """
        Create the index by checking for existing indexes, loading, or building a new one.
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
            self.documents = self.load_documents()
            self.build_index()

        except Exception as e:
            logger.exception(f"An error occurred during index creation: {e}")
            raise

    @abstractmethod
    def build_index(self):
        """
        Build the index from documents.
        """
        pass

    def save_index(self):
        """
        Save the index to persistent storage.
        """
        pass

    @abstractmethod
    def load_index(self):
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
