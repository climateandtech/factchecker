
from factchecker.indexing.base import BaseIndexer
from ragatouille import RAGPretrainedModel
from factchecker.tools.pdf_transformer import transform_pdf_to_txt
from llama_index.core import SimpleDirectoryReader

class ColBERTIndexer(BaseIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.index_name = self.options.pop('index_name', 'colbert_index') # Name of the index that will be built
        self.max_document_length = self.options.pop('max_document_length', 180) # The maximum length of a document. Documents longer than this will be split into chunks.
        self.split_documents = self.options.pop('split_documents', True) # Whether to split documents into chunks
        self.checkpoint = self.options.pop('checkpoint', 'colbert-ir/colbertv2.0') # The checkpoint of the pre-trained ColBERT model
    
    def load_and_transform_files(self):
        # Check if files are provided
        if not self.files:
            # Load files from source_directory using SimpleDirectoryReader
            self.files = SimpleDirectoryReader(self.source_directory).load_data()
        
        # Ensure documents are in text format
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
        # Load and transform the files to a list of Strings
        documents = self.load_and_transform_files()
        
        # Load the pre-trained ColBERT model
        rag = RAGPretrainedModel.from_pretrained(self.checkpoint)
        
        # Index the documents
        rag.index(
            collection=documents,
            index_name=self.index_name,
            max_document_length=self.max_document_length,
            split_documents=self.split_documents
        )
        print(f"Index created at .ragatouille/colbert/indexes/{self.index_name}")