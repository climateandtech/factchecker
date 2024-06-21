import os
from factchecker.indexing.base import BaseIndexer
from ragatouille import RAGPretrainedModel
from factchecker.tools.pdf_transformer import transform_pdf_to_txt
from llama_index.core import SimpleDirectoryReader

class ColBERTIndexer(BaseIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.index_name = self.options.pop('index_name', 'colbert_index')
        self.max_document_length = self.options.pop('max_document_length', 180)
        self.split_documents = self.options.pop('split_documents', True)
        self.checkpoint = self.options.pop('checkpoint', 'colbert-ir/colbertv2.0')
        self.index_path = f".ragatouille/colbert/indexes/{self.index_name}" # Default path to the indexes created by RAGatouille
        self.index_exists = self.check_index_exists()

    
    def check_index_exists(self):
        if not os.path.exists(self.index_path):
            return False

        # TODO: check if more checks are needed to ensure the index is valid

        return True

    def create_index(self):
        if self.index_exists:
            print(f"Index already exists at {self.index_path}")
            return

        rag = RAGPretrainedModel.from_pretrained(self.checkpoint)

        # Assuming `documents` is a list of llamaindex documents
        texts = [document.text for document in self.documents if hasattr(document, 'text')]

        rag.index(
            collection=texts,
            index_name=self.index_name,
            max_document_length=self.max_document_length,
            split_documents=self.split_documents
        )
        self.index_exists = True
        print(f"Index created at {self.index_path}")
