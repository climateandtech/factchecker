import os
import pytest
from factchecker.indexing.colbert_indexer import ColBERTIndexer
from factchecker.retrieval.colbert_retriever import ColBERTRetriever

@pytest.fixture
def setup_test_environment(tmp_path):
    # Create a temporary directory to act as the data source
    test_dir = tmp_path / "data"
    test_dir.mkdir()
    
    # Create dummy text files in the directory
    for i in range(5):
        with open(test_dir / f"test_file_{i}.txt", 'w') as f:
            f.write(f"This is the content of test file {i}.\n" * 10)
    
    # Return the path to the test directory
    return str(test_dir)

@pytest.fixture
def indexer_options(setup_test_environment):
    return {
        'source_directory': setup_test_environment,
        'index_name': 'colbert_pytest_index',
        'max_document_length': 180,
        'split_documents': True,
        'checkpoint': 'colbert-ir/colbertv2.0'
    }

def test_create_index(indexer_options):
    indexer = ColBERTIndexer(indexer_options)
    indexer.create_index()
    
    # Check if the index directory was created
    index_path = f".ragatouille/colbert/indexes/{indexer_options['index_name']}"
    assert os.path.exists(index_path), "Index directory was not created"
    
    # Check if the expected files are present in the index directory
    expected_files = [
        "0.codes.pt", "0.metadata.json", "0.residuals.pt", "avg_residual.pt",
        "buckets.pt", "centroids.pt", "collection.json",
        "doclens.0.json", "ivf.pid.pt", "metadata.json", "pid_docid_map.json", "plan.json"
    ]
    for filename in expected_files:
        assert os.path.exists(os.path.join(index_path, filename)), f"Missing expected file: {filename}"

def test_retrieve(indexer_options):
    indexer = ColBERTIndexer(indexer_options)
    indexer.create_index()  # Ensure the index is created before retrieving
    
    retriever = ColBERTRetriever(indexer)
    
    query = "content of test file"
    results = retriever.retrieve(query, top_k=5)
    
    # Check if the retriever returned results
    assert len(results) > 0, "No results returned from the retriever"
    
    # Check the content of the results
    for result in results:
        assert "content" in result, "Result does not contain 'content' key"
        assert "content of test file" in result['content'], "Result content does not match query"

# Run the tests
if __name__ == "__main__":
    pytest.main()
