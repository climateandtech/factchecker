import pytest
from llama_index.core import Document
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer

@pytest.fixture
def get_test_data_directory(tmp_path):
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
def get_test_documents():
    """Fixture to create a sequence of LlamaIndex Document objects from text strings."""
    texts = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]
    # Convert texts to Document objects
    documents = [Document(text=txt) for txt in texts]
    return documents


# @pytest.fixture
# def prepare_llama_vector_store_indexer(prepare_documents):
#     """Fixture to create and return a LlamaVectorStoreIndexer with indexed documents."""

#     indexer_options = {
#         'documents': prepare_documents,
#         'index_name': 'test_index_with_docs',
#     }
    
#     indexer = LlamaVectorStoreIndexer(indexer_options)
#     indexer.create_index()
    
#     return indexer

# @pytest.fixture
# def prepare_ragatouille_colbert_indexer(prepare_documents, tmp_path):

#     # Use the tmp_path fixture to create a temporary directory for the index
#     index_root = tmp_path / "indexes"

#     indexer_options = {
#         'documents': prepare_documents,
#         'index_name': 'test_index_with_docs',
#         'checkpoint': 'colbert-ir/colbertv2.0',
#         'index_root': str(index_root) 
#     }
    
#     indexer = RagatouilleColBERTIndexer(indexer_options)
#     indexer.create_index()

#     return indexer