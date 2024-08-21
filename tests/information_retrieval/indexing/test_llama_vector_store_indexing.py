import pytest
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer

def test_create_index_with_documents():
    # Prepare test documents
    documents = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]

    indexer_options = {
        'documents': documents,
        'index_name': 'test_index_with_docs',
    }
    
    indexer = LlamaVectorStoreIndexer(indexer_options)
    indexer.create_index()
    
    # Assertions
    assert indexer.index is not None, "Index should be created"
    assert indexer.documents is not None, "Documents should be loaded into the indexer"
    assert len(indexer.documents) == len(documents), "All documents should be indexed"

def test_create_index_from_directory(prepare_test_data_directory):
    # Using the prepared test data directory
    indexer_options = {
        'source_directory': prepare_test_data_directory,
        'index_name': 'test_index_from_dir',
    }
    
    indexer = LlamaVectorStoreIndexer(indexer_options)
    indexer.create_index()
    
    # Assertions
    assert indexer.index is not None, "Index should be created"
    assert indexer.documents is not None, "Documents should be loaded from the directory"
    assert len(indexer.documents) == 5, "There should be 5 documents indexed"

def test_create_index_when_index_already_exists():
    documents = [
        "This is a test document for the existing index."
    ]
    
    indexer_options = {
        'documents': documents,
        'index_name': 'test_existing_index',
    }

    indexer = LlamaVectorStoreIndexer(indexer_options)
    indexer.create_index()
    
    # Attempt to create the index again
    indexer.create_index()
    
    # Assertions
    assert indexer.index is not None, "Index should still exist"
    assert indexer.documents is not None, "Documents should still be loaded into the indexer"
    assert len(indexer.documents) == len(documents), "The same number of documents should be indexed"

# TODO: Add more tests as needed for additional methods and edge cases
