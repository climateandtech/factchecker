"""Tests for the LlamaVectorStoreIndexer class."""

import pytest
from llama_index.core import Document

from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer


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

@pytest.mark.integration
def test_initialize_index_from_directory(get_test_data_directory: str) -> None:
    """Initialize an index with a directory containing text files."""
    indexer_options = {
        'source_directory': get_test_data_directory,
        'index_name': 'test_index_from_dir',
    }

    indexer = LlamaVectorStoreIndexer(indexer_options)
    assert indexer.index is None
    indexer.initialize_index()
    assert indexer.index is not None
    assert indexer.index_name == 'test_index_from_dir'
