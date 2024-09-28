from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from llama_index.core import SimpleDirectoryReader

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

    def load_documents(self):
        """
        Load documents either from preloaded data or from the specified source.

        Returns:
            List[Any]: A list of documents loaded for indexing.
        """
        try:
            if self.documents:
                return self.documents
            if not self.files:
                # Load files from the source directory if no files are provided in the options
                return SimpleDirectoryReader(self.source_directory).load_data()
            else:
                # Load files from the provided list of files
                return SimpleDirectoryReader(input_files=self.files).load_data()
        except Exception as e:
            print(f"An error occurred while loading documents: {e}")
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
        self.documents = self.load_documents()

        # Check if the index already exists
        if self.index is not None:
            print("Index object already exists. Skipping index creation.")
            return True  # Indicate that the index already exists
        
        # Load index if it exists on disk
        if self.check_persisted_index_exists():
            print(f"Saved index found at {self.index_path}. Loading index...")
            self.load_index()
            return True  # Indicate that the index was loaded

        return False  # Indicate that the index needs to be created

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    def add_to_index(self, documents):
        pass

    @abstractmethod
    def delete_from_index(self, document_ids):
        pass
