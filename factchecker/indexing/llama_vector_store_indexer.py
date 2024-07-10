# vector_store_indexer.py
from abstract_indexer import AbstractIndexer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

class LlamaVectorStoreIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.max_document_length = self.options.get('max_document_length', 180)
        self.split_documents = self.options.get('split_documents', True)
        self.embedding_model = self.options.pop('embedding_model', None)
        self.vector_store = self.options.pop('vector_store', None)
        self.transformations = self.options.pop('transformations', [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)])
        self.index = None

    def check_index_exists(self):
        return self.index is not None

    def create_index(self, documents):
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store, **self.options)
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embedding_model,
            transformations=self.transformations
        )
        print(f"Index created: {self.index_name}")

    def add_document_to_index(self, document):
        print(f"Adding document to Llama Vector Store index: {self.index_name}")
        self.index.insert(document)

    def remove_document_from_index(self, document_id):
        print(f"Removing document from Llama Vector Store index: {self.index_name}")
        self.index.delete(document_id)

    def update_document_in_index(self, document):
        print(f"Updating document in Llama Vector Store index: {self.index_name}")
        self.remove_document_from_index(document.id)
        self.add_document_to_index(document)
