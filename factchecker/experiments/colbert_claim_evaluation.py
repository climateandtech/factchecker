from factchecker.indexing.colbert_indexer import ColBERTIndexer
from factchecker.retrieval.colbert_retriever import ColBERTRetriever

def main():
    indexer_options = {
        'source_directory': 'data',
        'index_name': 'colbert_index',
        'max_document_length': 180,
        'split_documents': True,
        'checkpoint': 'colbert-ir/colbertv2.0'
    }
    retriever_options = {
    }

    indexer = ColBERTIndexer(indexer_options)
    retriever = ColBERTRetriever(indexer, retriever_options)

    query = "Climate change would have happened without humans"

    results = retriever.retrieve(query, top_k=5)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
