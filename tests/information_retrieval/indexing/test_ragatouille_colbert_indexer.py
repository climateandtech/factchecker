"""Tests for the RagatouilleColBERTIndexer class."""

import os
from pathlib import Path

import pytest
from llama_index.core import Document

from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer


@pytest.mark.integration
def test_initialize_index_from_documents(get_test_documents: list[Document], tmp_path: str) -> None:
    """Initialize an index with a list of documents."""
    # Use the tmp_path fixture to create a temporary directory for the index
    index_root = tmp_path / "indexes/ragatouille"

    indexer_options = {
        'documents': get_test_documents,
        'index_name': 'test_index_from_docs',
        'index_root': str(index_root) 
    }
    
    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.initialize_index()
    
    # Assertions
    assert indexer.index is not None
    assert indexer.index_path is not None
    assert os.path.exists(indexer.index_path)

@pytest.mark.integration
def test_initialize_index_from_directory(get_test_data_directory: str, tmp_path: Path) -> None:
    """Initialize an index with a directory of text files."""
    # Use the tmp_path fixture to create a temporary directory for the index
    index_root = tmp_path / "indexes/ragatouille"

    indexer_options = {
        'source_directory': get_test_data_directory,
        'index_name': 'test_index_from_dir',
        'index_root': str(index_root) 
    }
    
    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.initialize_index()
    
    # Assertions
    assert indexer.index is not None
    assert indexer.index_path is not None
    assert os.path.exists(indexer.index_path)

@pytest.mark.integration
def test_load_existing_index(get_ragatouille_colbert_indexer: RagatouilleColBERTIndexer, tmp_path: Path) -> None:
    """Test loading an existing index from disk."""
    indexer = get_ragatouille_colbert_indexer
    
    # Save the current index path for later comparison
    saved_index_path = indexer.index_path
    
    # Create a new instance of the indexer with the same index path
    indexer_options = {
        'index_name': 'test_loaded_index_from_docs',
        'index_path': saved_index_path,
    }
    
    new_indexer = RagatouilleColBERTIndexer(indexer_options)
    new_indexer.load_index()
    
    # Assertions
    assert new_indexer.index is not None
    assert new_indexer.index_path == saved_index_path
    assert new_indexer.index_name == 'test_loaded_index_from_docs'