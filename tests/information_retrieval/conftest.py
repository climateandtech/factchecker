from pathlib import Path

import pytest
from llama_index.core import Document

from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from unittest.mock import patch


@pytest.fixture
def get_test_data_directory(tmp_path: Path) -> str:
    """Creates a temporary directory with dummy text files for indexing tests."""
    # Create a temporary directory to act as the data source
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create dummy text files in the directory
    for i in range(5):
        with open(data_dir / f"test_file_{i}.txt", 'w') as f:
            f.write(f"This is the content of test file {i}.\n" * 10)
    
    # Return the path to the test directory
    return str(data_dir)


@pytest.fixture
def get_test_documents() -> list[Document]:
    """Fixture to create a sequence of LlamaIndex Document objects from text strings."""
    texts = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]
    # Convert texts to Document objects
    documents = [Document(text=txt) for txt in texts]
    return documents

@pytest.fixture
def get_many_test_documents() -> list[Document]:
    """Create a large list of dummy Document objects for testing."""
    documents = []
    for i in range(15000):
        documents.append(Document(text=f"This is a longer test document with the number {i}."))
    return documents


@pytest.fixture
def get_llama_vector_store_indexer(get_test_documents: list[Document]) -> LlamaVectorStoreIndexer:
    """Fixture to create and return a LlamaVectorStoreIndexer with indexed documents."""
    with patch.dict('os.environ', {
        'EMBEDDING_TYPE': 'mock',
        'MOCK_EMBED_DIM': '384'
    }):
        indexer_options = {
            'documents': get_test_documents,
            'index_name': 'test_index_with_docs',
            'transformations': [],  # Disable transformations to keep documents intact
        }

        indexer = LlamaVectorStoreIndexer(indexer_options)
        # indexer.initialize_index()
        return indexer


@pytest.fixture
def get_ragatouille_colbert_indexer(get_many_test_documents: list[Document], tmp_path: Path) -> RagatouilleColBERTIndexer:
    """Fixture to create and return a RagatouilleColBERTIndexer with indexed documents."""
    # Use the tmp_path fixture to create a temporary directory for the index
    index_root = tmp_path / "indexes/ragatouille"

    indexer_options = {
        'documents': get_many_test_documents,
        'index_name': 'test_index_from_docs',
        'index_root': str(index_root) 
    }
    
    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.initialize_index()

    return indexer