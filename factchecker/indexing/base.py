from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

class BaseIndexer:
    def __init__(self, options=None):
        self.options = options if options is not None else {}
        # Extract and remove 'source_directory' from options if it exists
        self.source_directory = self.options.pop('source_directory', 'data')
        self.files = self.options.pop('files', None) or SimpleDirectoryReader(self.source_directory).load_data()
        self.embedding_model = self.options.pop('embedding_model', None)
        self.vector_store = self.options.pop('vector_store', None)
        self.transformations = self.options.pop('transformations', [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)])

    def create_index(self):
        # Now self.options will only contain relevant options for StorageContext.from_defaults
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store, **self.options)
        self.index = VectorStoreIndex.from_documents(
            self.files, 
            storage_context=storage_context, 
            embed_model=self.embedding_model,
            transformations=self.transformations
        )