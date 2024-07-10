# vector_store_indexer.py
from abstract_indexer import AbstractIndexer
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

class LlamaVectorStoreIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.embedding_model = self.options.pop('embedding_model', None)
        self.vector_store = self.options.pop('vector_store', None)
        self.transformations = self.options.pop('transformations', [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)])
        self.show_progress = self.options.pop('show_progress', False)
        self.index = None

    def check_index_exists(self):
        return self.index is not None

    def create_index(self):
        # Now self.options will only contain relevant options for StorageContext.from_defaults
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store, **self.options)
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            embed_model=self.embedding_model,
            transformations=self.transformations
        )
        print(f"Index created: {self.index_name}")

    def insert_document_to_index(self, document):
        print(f"Adding document to Llama Vector Store index: {self.index_name}")
        self.index.insert(document)

    def delete_document_from_index(self, document_id):
        print(f"Removing document from Llama Vector Store index: {self.index_name}")
        self.index.delete(document_id)


# ---- testing

# indexer_options = {
#     'source_directory': '../../data',
#     'index_name': 'vector_store_test_index',
#     'show_progress': True
# }
# indexer = LlamaVectorStoreIndexer(indexer_options)
# indexer.create_index()
# retriever = indexer.index.as_retriever()
# retriever.retrieve("carbon emissions")