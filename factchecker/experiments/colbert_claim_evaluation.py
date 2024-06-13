from factchecker.indexing.colbert_indexer import ColBERTIndexer

def main():
    # Define the options for each component
    indexer_options = {
        'source_directory': 'data',
        'index_name': 'colbert_index',
        'max_document_length': 180,
        'split_documents': True,
        'checkpoint': 'colbert-ir/colbertv2.0'
    }

    # Initialize the ColBERT indexer and create the index
    indexer = ColBERTIndexer(indexer_options)
    indexer.create_index()

if __name__ == "__main__":
    main()