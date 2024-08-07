from .abstract_indexer import AbstractIndexer
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
        # TODO: check if a saved index exists and load it instead of creating a new one

        self.index = ColbertIndex.from_documents(
            self.documents,
            gpus=self.gpus,
            show_progress=self.show_progress,
            # index_name=self.index_name,
            )

        # TODO: save index to disk

    def add_to_index(self, documents):
        # TODO: implement
        pass

    def delete_from_index(self, document_ids):
        # TODO: implement
        pass


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