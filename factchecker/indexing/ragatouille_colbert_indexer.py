# ragatouille_colbert_indexer.py
import os
from .abstract_indexer import AbstractIndexer
from ragatouille import RAGPretrainedModel

class RagatouilleColBERTIndexer(AbstractIndexer):
    def __init__(self, options=None):
        super().__init__(options)
        self.max_document_length = self.options.pop('max_document_length', 256)
        self.checkpoint = self.options.pop('checkpoint', 'colbert-ir/colbertv2.0')
        self.index_path = f".ragatouille/colbert/indexes/{self.index_name}" # Default path to the indexes created by RAGatouille

    def check_index_exists(self):
        return os.path.exists(self.index_path)

    def create_index(self):
        if self.check_index_exists():
            print(f"Index already exists at {self.index_path}")
            return

        rag = RAGPretrainedModel.from_pretrained(self.checkpoint)

        # Extract text from Document objects
        texts = [document.text for document in self.documents if hasattr(document, 'text')]

        self.index = rag.index(
            collection=texts,
            index_name=self.index_name,
            max_document_length=self.max_document_length,
        )
        print(f"Index created at {self.index_path}")

    def insert_document_to_index(self, document):
        # print(f"Adding document to ColBERT index: {self.index_name}")
        # # Assuming `document` is a text string
        # rag = RAGPretrainedModel.from_pretrained(self.checkpoint)
        # rag.add(document, index_name=self.index_name)
        pass # TODO: implement

    def delete_document_from_index(self, document_id):
        # print(f"Removing document from ColBERT index: {self.index_name}")
        # # Assuming `document_id` is the identifier of the document to be removed
        # rag = RAGPretrainedModel.from_pretrained(self.checkpoint)
        # rag.remove(document_id, index_name=self.index_name)
        pass # TODO: implement

# ---- quick testing in notebook mode


def main():

    ColbertIndexer = RagatouilleColBERTIndexer().create_index()


if __name__ == '__main__':
    main()