import pytest
from llama_index.core import Document

@pytest.fixture
def prepare_test_data_directory(tmp_path):
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
def prepare_documents():
    """Fixture to create a sequence of LlamaIndex Document objects from text strings."""
    texts = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]
    # Convert texts to Document objects
    documents = [Document(text=txt) for txt in texts]
    return documents