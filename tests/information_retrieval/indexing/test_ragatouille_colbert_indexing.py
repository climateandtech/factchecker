import pytest
from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer

# def test_create_index_with_documents(prepare_documents):
#     indexer_options = {
#         'documents': prepare_documents,
#         'index_name': 'test_index_with_docs',
#         'checkpoint': 'colbert-ir/colbertv2.0'
#     }
    
#     indexer = RagatouilleColBERTIndexer(indexer_options)
#     indexer.create_index()
    
#     # Assertions
#     assert indexer.index is not None, "Index should be created"
#     assert indexer.documents is not None, "Documents should be loaded into the indexer"
#     assert len(indexer.documents) == len(prepare_documents), "All documents should be indexed"
#     assert indexer.index_path is not None, "Index path should be set after creating the index"

# def test_create_index_from_directory(prepare_test_data_directory):
#     indexer_options = {
#         'source_directory': prepare_test_data_directory,
#         'index_name': 'test_index_from_dir',
#         'checkpoint': 'colbert-ir/colbertv2.0'
#     }
    
#     indexer = RagatouilleColBERTIndexer(indexer_options)
#     indexer.create_index()
    
#     # Assertions
#     assert indexer.index is not None, "Index should be created"
#     assert indexer.documents is not None, "Documents should be loaded from the directory"
#     assert len(indexer.documents) == 5, "There should be 5 documents indexed"
#     assert indexer.index_path is not None, "Index path should be set after creating the index"

# def test_load_existing_index(mocker):
#     """Test loading an existing index from disk."""
#     mocker.patch('ragatouille.RAGPretrainedModel.from_index', return_value="LoadedIndex")
    
#     indexer_options = {
#         'index_name': 'existing_test_index',
#         'index_path': '/path/to/existing/index',
#         'checkpoint': 'colbert-ir/colbertv2.0'
#     }
    
#     indexer = RagatouilleColBERTIndexer(indexer_options)
#     indexer.load_index()
    
#     # Assertions
#     assert indexer.index == "LoadedIndex", "Index should be loaded from disk"
#     assert indexer.index_path == '/path/to/existing/index', "Index path should match the provided path"

def test_create_index_when_index_already_exists(mocker, prepare_documents):
    """Test that the index creation is skipped if the index already exists."""
    mocker.patch.object(RagatouilleColBERTIndexer, 'check_persisted_index_exists', return_value=True)
    mocker.patch('ragatouille.RAGPretrainedModel.from_index', return_value="LoadedIndex")

    indexer_options = {
        'documents': prepare_documents,
        'index_name': 'test_existing_index',
        'index_path': '/path/to/existing/index',
        'checkpoint': 'colbert-ir/colbertv2.0'
    }

    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.create_index()
    
    # Assertions
    assert indexer.index == "LoadedIndex", "Existing index should be loaded instead of creating a new one"
    assert indexer.index_path == '/path/to/existing/index', "Index path should match the provided path"

# # Add more tests as needed for the add_to_index and delete_from_index methods
