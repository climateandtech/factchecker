from unittest.mock import patch
import numpy as np
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
import pytest
from llama_index.core import Document

@patch('llama_index.embeddings.openai.OpenAIEmbedding')
def test_initialize_index_from_documents(mock_embedding, get_test_documents):
    # Configure mock to return fixed embeddings
    mock_embedding.return_value._get_text_embedding.return_value = np.array([0.1] * 384)
    mock_embedding.return_value._get_query_embedding.return_value = np.array([0.1] * 384)

@pytest.mark.integration
def test_initialize_index_from_documents(get_test_documents: list[Document]) -> None:
    """Initialize an index with a list of documents."""
    indexer_options = {
        'documents': get_test_documents,
        'index_name': 'test_index_with_docs',
    }

    indexer = LlamaVectorStoreIndexer(indexer_options)
    assert indexer.index is None
    indexer.initialize_index()
    
    assert indexer.initial_documents == get_test_documents
    assert indexer.index is not None
    assert indexer.index_name == 'test_index_with_docs'

@patch('llama_index.embeddings.openai.OpenAIEmbedding')
def test_initialize_index_from_directory(mock_embedding, get_test_data_directory):
    # Configure mock to return fixed embeddings
    mock_embedding.return_value._get_text_embedding.return_value = np.array([0.1] * 384)
    mock_embedding.return_value._get_query_embedding.return_value = np.array([0.1] * 384)

    indexer_options = {
        'source_directory': get_test_data_directory,
        'index_name': 'test_index_from_dir',
    }

    indexer = LlamaVectorStoreIndexer(indexer_options)
    assert indexer.index is None
    indexer.initialize_index()
    assert indexer.index is not None
    assert indexer.index_name == 'test_index_from_dir'

@patch('llama_index.embeddings.openai.OpenAIEmbedding')
def test_save_and_load_index(mock_embedding, get_test_documents, tmp_path):
    """Test saving and loading an index"""
    # Configure mock to return fixed embeddings
    mock_embedding.return_value._get_text_embedding.return_value = np.array([0.1] * 384)
    mock_embedding.return_value._get_query_embedding.return_value = np.array([0.1] * 384)
    
    # Initialize indexer with a temporary path
    storage_path = str(tmp_path / "test_index")
    indexer_options = {
        'documents': get_test_documents,
        'index_name': 'test_save_load',
        'storage_path': storage_path
    }
    
    # Create and save index
    indexer = LlamaVectorStoreIndexer(indexer_options)
    indexer.initialize_index()
    indexer.save_index()
    
    # Create new indexer and load saved index
    new_indexer = LlamaVectorStoreIndexer(indexer_options)
    new_indexer.load_index()
    
    # Verify index was loaded successfully
    assert new_indexer.index is not None
    assert new_indexer.index_name == 'test_save_load'
    
    # Test some index operations to verify functionality
    query_text = "test query"
    original_results = indexer.index.as_query_engine().query(query_text)
    loaded_results = new_indexer.index.as_query_engine().query(query_text)
    assert str(original_results) == str(loaded_results)
