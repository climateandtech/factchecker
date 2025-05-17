from factchecker.indexing.abstract_indexer import AbstractIndexer

class TestIndexer(AbstractIndexer):
    def build_index(self, documents):
        self.documents = documents
    
    def add_to_index(self, documents):
        self.documents.extend(documents)
    
    def delete_from_index(self, document_ids):
        pass