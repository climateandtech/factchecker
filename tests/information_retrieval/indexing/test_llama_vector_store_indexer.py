"""Tests for the LlamaVectorStoreIndexer class."""

import os

import pytest
from llama_index.core import Document
from unittest.mock import patch

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


# --- save_index / load_index (TDD) ---
@pytest.mark.integration
def test_save_index_persists_to_disk(get_test_documents: list[Document], tmp_path) -> None:
    """Build index, call save_index, assert index_path contains expected files."""
    index_path = str(tmp_path / "index")
    with patch.dict('os.environ', {'EMBEDDING_TYPE': 'mock', 'MOCK_EMBED_DIM': '384'}):
        indexer_options = {
            'documents': get_test_documents,
            'index_name': 'test_save',
            'index_path': index_path,
            'transformations': [],
        }
        indexer = LlamaVectorStoreIndexer(indexer_options)
        indexer.build_index(get_test_documents)
        indexer.save_index()
    assert os.path.isdir(index_path)
    # LlamaIndex persists docstore, index_store, vector store, graph store
    assert os.path.isfile(os.path.join(index_path, 'docstore.json')) or os.path.exists(
        os.path.join(index_path, 'default__vector_store.json')
    )


@pytest.mark.integration
def test_load_index_restores(get_test_documents: list[Document], tmp_path) -> None:
    """Build and save; create new indexer, load_index; assert index is usable (retriever returns nodes)."""
    index_path = str(tmp_path / "index")
    with patch.dict('os.environ', {'EMBEDDING_TYPE': 'mock', 'MOCK_EMBED_DIM': '384'}):
        indexer_options = {
            'documents': get_test_documents,
            'index_name': 'test_load',
            'index_path': index_path,
            'transformations': [],
        }
        indexer = LlamaVectorStoreIndexer(indexer_options)
        indexer.build_index(get_test_documents)
        indexer.save_index()
    # Load in a new indexer (same index_path, no documents)
    with patch.dict('os.environ', {'EMBEDDING_TYPE': 'mock', 'MOCK_EMBED_DIM': '384'}):
        indexer2 = LlamaVectorStoreIndexer({
            'index_name': 'test_load',
            'index_path': index_path,
        })
        indexer2.load_index()
    assert indexer2.index is not None
    retriever = indexer2.index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("first document")
    assert len(nodes) >= 1
