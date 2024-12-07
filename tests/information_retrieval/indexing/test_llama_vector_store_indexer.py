
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer

def test_initialize_index_from_documents(get_test_documents):

    indexer_options = {
        'documents':get_test_documents,
        'index_name': 'test_index_with_docs',
    }
    
    indexer = LlamaVectorStoreIndexer(indexer_options)
    indexer.initialize_index()
    
    assert indexer.index is not None
    assert indexer.index_name == 'test_index_with_docs'

def test_initialize_index_from_directory(get_test_data_directory):

    indexer_options = {
        'source_directory': get_test_data_directory,
        'index_name': 'test_index_from_dir',
    }
    
    indexer = LlamaVectorStoreIndexer(indexer_options)
    indexer.initialize_index()
    
    assert indexer.index is not None
    assert indexer.index_name == 'test_index_from_dir'
