from factchecker.indexing.abstract_indexer import AbstractIndexer
from llama_index.indices.managed.colbert import ColbertIndex


class LlamaColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.gpus = self.options.pop('gpus', '0')
        self.show_progress = self.options.pop('show_progress', False)
        # TODO: check LlamaIndex documentation for more relevant parameters to pass to the constructor instead of the default values

    def check_persisted_index_exists(self):
        # TODO: first implement the methods to save and load this type of index
        # TODO: then implement this method to check if the index exists on disk
        pass

    def create_index(self):
        
        # Stop further execution if the index already exists or was loaded
        if super().create_index():
            return  
        
        # There is currently (28.9.2024) a bug with the LlamaIndex ColBERT Indexing using the from_documents function resulting in:
        # TypeError: ColbertIndex._build_index_from_nodes() got an unexpected keyword argument 'index_name'
        # See: https://github.com/run-llama/llama_index/issues/14398 

        self.index = ColbertIndex.from_documents(
            self.documents,
            gpus=self.gpus,
            show_progress=self.show_progress,
            )

    def load_index(self):
        raise NotImplementedError("load_index() of LlamaColBERTIndexer is not yet implemented")
        pass

    def add_to_index(self, documents):
        raise NotImplementedError("add_to_index() of LlamaColBERTIndexer is not yet implemented")
        pass

    def delete_from_index(self, document_ids):
        raise NotImplementedError("delete_from_index() of LlamaColBERTIndexer is not yet implemented")
        pass