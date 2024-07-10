# colbert_indexer.py
import os
from abstract_indexer import AbstractIndexer
from llama_index.indices.managed.colbert import ColbertIndex
from llama_index.core.schema import TextNode


class LlamaColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.index_path = f".llama_index/colbert/indexes/{self.index_name}"
        self.model_name = self.options.pop('model_name', 'colbert-ir/colbertv2.0')
        self.gpus = self.options.pop('gpus', '0')
        # TODO: check LlamaIndex documentation for more relevant parameters to pass to the constructor instead of the default values

    def check_index_exists(self):
        return os.path.exists(self.index_path)

    def create_index(self):
        if self.check_index_exists():
            print(f"Index already exists at {self.index_path}")
            return
        
        # Assuming `documents` is a list of llamaindex documents
        # nodes = [TextNode(text=doc.get_text(), id_=doc.get_id()) for doc in self.documents]

        self.index = ColbertIndex(
                nodes=self.documents,
                index_path=self.index_path,
                model_name=self.model_name,
                gpus=self.gpus
            )

        # self.index = ColbertIndex.from_documents(self.documents)
     
        print(f"Index created at {self.index_path}")

    def insert_document_to_index(self, document):
        print(f"Adding document to Llama ColBERT index: {self.index_name}")
        self.index.add(document)

    def delete_document_from_index(self, document_id):
        print(f"Removing document from Llama ColBERT index: {self.index_name}")
        self.index.remove(document_id)


# ---- testing


indexer_options = {
    'source_directory': '../../data',
    'index_name': 'colbert_test_index',
}

indexer = LlamaColBERTIndexer(indexer_options)

indexer.create_index()