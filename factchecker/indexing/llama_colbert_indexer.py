# colbert_indexer.py
import os
from .abstract_indexer import AbstractIndexer
from llama_index.indices.managed.colbert import ColbertIndex


class LlamaColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.gpus = self.options.pop('gpus', '0')
        self.show_progress = self.options.pop('show_progress', False)
        # TODO: check LlamaIndex documentation for more relevant parameters to pass to the constructor instead of the default values

    def check_index_exists(self):
        return self.index is not None

    def create_index(self):
        if self.check_index_exists():
            print(f"Index already exists at {self.index_path}")
            return

        self.index = ColbertIndex.from_documents(
            self.documents,
            gpus=self.gpus,
            show_progress=self.show_progress,
            index_name=self.index_name,
            )
     
        print(f"Index created at {self.index.index_path}")

    def insert_document_to_index(self, document):
        #print(f"Adding document to Llama ColBERT index: {self.index_name}")
        self.index.add(document)

    def delete_document_from_index(self, document_id):
        #print(f"Removing document from Llama ColBERT index: {self.index_name}")
        self.index.remove(document_id)


# # ---- quick testing 

indexer_options = {
    'index_name': 'quick_test_llama_colbert_index',
    'source_directory': 'data',
    'show_progress': True,
}
indexer = LlamaColBERTIndexer()
indexer.create_index()
retriever = indexer.index.as_query_engine(similarity_top_k=3)
response = retriever.retrieve('climate change is real and caused by humans')
print(f'Retrieved documents: {response}')