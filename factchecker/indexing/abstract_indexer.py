from abc import ABC, abstractmethod
from llama_index.core import SimpleDirectoryReader

class AbstractIndexer(ABC):
    def __init__(self, options=None):
        self.options = options if options is not None else {}
        self.index_name = self.options.pop('index_name', 'default_index')
        self.index_path = self.options.pop('index_path', None) # Path to the directory where the index is stored on disk
        self.source_directory = self.options.pop('source_directory', 'data')
        self.files = self.options.pop('files', None)
        self.documents = self.options.pop('documents', None) # List of LLamaIndex Documents as returned by SimpleDirectoryReader
        self.index = None  # The in-memory index object

    def load_documents(self):
        if self.documents:
            return self.documents
        if not self.files:
            # Load files from the source directory if no files are provided in the options
            return SimpleDirectoryReader(self.source_directory).load_data()
        else:
            # Load files from the provided list of files
            return SimpleDirectoryReader(input_files=self.files).load_data()
        
    @abstractmethod
    def check_persisted_index_exists(self):
        pass

    @abstractmethod
    def create_index(self):
        self.documents = self.load_documents()

        if self.index is not None:
            print("Index object already exists. Skipping index creation.")
            return True  # Indicate that the index already exists
        
        # Load index if it exists on disk
        if self.check_persisted_index_exists():
            print(f"Saved index found at {self.index_path}. Loading index...")
            self.load_index()
            return True  # Indicate that the index was loaded

        return False  # Indicate that the index needs to be created

    # method to load index. for future implementation
    @abstractmethod
    def load_index(self):
        pass

    # # method to save index. for future implementation
    # @abstractmethod
    # def persist_index(self):
    #     pass

    @abstractmethod
    def add_to_index(self, documents):
        pass

    @abstractmethod
    def delete_from_index(self, document_ids):
        pass
