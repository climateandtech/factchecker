# abstract_indexer.py
from abc import ABC, abstractmethod
from llama_index.core import SimpleDirectoryReader

class AbstractIndexer(ABC):
    def __init__(self, options=None):
        self.options = options if options is not None else {}
        self.index_name = self.options.get('index_name', 'default_index')
        self.source_directory = self.options.pop('source_directory', 'data')
        self.files = self.options.pop('files', None)
        self.documents = self.load_documents()

    def load_documents(self):
        if not self.files:
            # Load files from the source directory if no files are provided in the options
            return SimpleDirectoryReader(self.source_directory).load_data()
        else:
            # Load files from the provided list of files
            return SimpleDirectoryReader(input_files=self.files).load_data()

    @abstractmethod
    def create_index(self, documents):
        pass

    @abstractmethod
    def check_index_exists(self):
        pass

    @abstractmethod
    def add_document_to_index(self, document):
        pass

    @abstractmethod
    def remove_document_from_index(self, document_id):
        pass

    @abstractmethod
    def update_document_in_index(self, document):
        pass