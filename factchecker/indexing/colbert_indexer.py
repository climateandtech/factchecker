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
        self.index_path = f".ragatouille/colbert/indexes/{self.index_name}"
        self.index = self.index_path if os.path.exists(self.index_path) else None

    def load_and_transform_files(self):
        if not self.files:
            self.files = SimpleDirectoryReader(self.source_directory).load_data()

        documents = []
        for file in self.files:
            if isinstance(file, str) and file.endswith('.pdf'):
                documents.append(transform_pdf_to_txt(file))
            elif isinstance(file, str) and file.endswith('.txt'):
                with open(file, 'r') as f:
                    documents.append(f.read())
            elif hasattr(file, 'text'):
                documents.append(file.text)
            else:
                print(f"Unsupported file format: {file}")
        
        return documents

    def create_index(self):
        if self.index is not None:
            print(f"Index already exists at {self.index_path}")
            return

        documents = self.load_and_transform_files()
        rag = RAGPretrainedModel.from_pretrained(self.checkpoint)
        rag.index(
            collection=documents,
            index_name=self.index_name,
            max_document_length=self.max_document_length,
            split_documents=self.split_documents
        )
        self.index = self.index_path
        print(f"Index created at {self.index_path}")
