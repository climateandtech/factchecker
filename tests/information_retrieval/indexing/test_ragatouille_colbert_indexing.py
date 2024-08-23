import pytest
import os
from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer

def test_create_index_with_documents(prepare_documents, tmp_path):

    # Use the tmp_path fixture to create a temporary directory for the index
    index_root = tmp_path / "indexes"

    indexer_options = {
        'documents': prepare_documents,
        'index_name': 'test_index_with_docs',
        'checkpoint': 'colbert-ir/colbertv2.0',
        'index_root': str(index_root) 
    }
    
    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.create_index()
    
    # Assertions
    assert indexer.index is not None, "Index should be created"
    assert indexer.documents is not None, "Documents should be loaded into the indexer"
    assert len(indexer.documents) == len(prepare_documents), "All documents should be indexed"
    assert indexer.index_path is not None, "Index path should be set after creating the index"
    assert os.path.exists(indexer.index_path), "Index should be saved to disk in the specified path"

def test_create_index_from_directory(prepare_test_data_directory, tmp_path):

    # Use the tmp_path fixture to create a temporary directory for the index
    index_root = tmp_path / "indexes"
    
    indexer_options = {
        'source_directory': prepare_test_data_directory,
        'index_name': 'test_index_from_dir',
        'checkpoint': 'colbert-ir/colbertv2.0',
        'index_root': str(index_root)
    }
    
    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.create_index()
    
    # Assertions
    assert indexer.index is not None, "Index should be created"
    assert indexer.documents is not None, "Documents should be loaded from the directory"
    assert len(indexer.documents) == 5, "There should be 5 documents indexed"
    assert indexer.index_path is not None, "Index path should be set after creating the index"
    assert os.path.exists(indexer.index_path), "Index should be saved to disk in the specified path"


# # TODO
# def test_load_existing_index(mocker):   
#     pass

# # TODO
# def test_create_index_when_index_already_exists(mocker, prepare_documents):
#     pass

# # Add more tests as needed for the add_to_index and delete_from_index methods
