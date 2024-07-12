# colbert_indexer.py
import os
from .abstract_indexer import AbstractIndexer
from llama_index.indices.managed.colbert import ColbertIndex


class LlamaColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.index_path = f".llama_index/colbert/indexes/{self.index_name}"
        self.gpus = self.options.pop('gpus', '0')
        # TODO: check LlamaIndex documentation for more relevant parameters to pass to the constructor instead of the default values

    def check_index_exists(self):
        return os.path.exists(self.index_path)

    def create_index(self):
        if self.check_index_exists():
            print(f"Index already exists at {self.index_path}")
            return

        self.index = ColbertIndex.from_documents(self.documents)
     
        print(f"Index created at {self.index}")

    def insert_document_to_index(self, document):
        print(f"Adding document to Llama ColBERT index: {self.index_name}")
        self.index.add(document)

    def delete_document_from_index(self, document_id):
        print(f"Removing document from Llama ColBERT index: {self.index_name}")
        self.index.remove(document_id)


# ---- quick testing 

indexer = LlamaColBERTIndexer()
indexer.create_index()

# colbert_indexer.py
# import os
# from llama_index.indices.managed.colbert import ColbertIndex
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core.schema import TextNode

# def main():
#     # source_directory = '../../data'
#     source_directory = 'data'

#     print("Loading documents...")
#     documents = SimpleDirectoryReader(source_directory).load_data()
#     print(f"Loaded {len(documents)} documents.")

#     try:
#         # print("Creating ColBERT index...")
#         # index = ColbertIndex.from_documents(
#         #     documents,
#         #     index_name="colbert_test_index",
#         #     model_name="colbert-ir/colbertv2.0",
#         #     gpus=0,  # Set to the appropriate number of GPUs
#         #     show_progress=True,
#         # )
#         # print("ColBERT index created successfully.")

#         print("Creating ColBERT index...")
#         # nodes = [TextNode(text=doc.get_text(), id_=doc.get_id()) for doc in documents]
#         nodes = [TextNode(text=doc.text, id_=doc.doc_id) for doc in documents]

#         index = ColbertIndex(
#                 nodes=nodes
#             )
#         print("ColBERT index created successfully.")

#         # # Check if the index directory contains the expected files
#         # if os.path.exists("storage/colbert_index/colbert_test_index"):
#         #     print("Index files are successfully created.")
#         # else:
#         #     print("Index creation seems to have failed.")

#     except Exception as e:
#         print(f"Error during indexing: {e}")

# if __name__ == '__main__':
#     main()
