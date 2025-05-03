import pytest
from pathlib import Path
import json
import time
from llama_index.core import Document
from factchecker.indexing.abstract_indexer import AbstractIndexer
from .conftest import TestIndexer

def test_abstract_class_enforcement():
    """Test that abstract methods must be implemented"""
    
    # This should raise TypeError because abstract methods aren't implemented
    with pytest.raises(TypeError):
        class IncompleteIndexer(AbstractIndexer):
            pass
        
        IncompleteIndexer()

    # This should work because all abstract methods are implemented
    class CompleteIndexer(AbstractIndexer):
        def build_index(self, documents):
            pass
        
        def add_to_index(self, documents):
            pass
        
        def delete_from_index(self, document_ids):
            pass
    
    indexer = CompleteIndexer({'index_name': 'test'})
    assert isinstance(indexer, AbstractIndexer)

def test_metadata_creation(tmp_path):
    """Test that metadata is created correctly with default values"""
    options = {
        'index_name': 'test_index',
        'storage_path': str(tmp_path / "indexes/test_index")
    }
    
    indexer = TestIndexer(options)
    
    # Check metadata file exists and directory was created
    metadata_path = Path(tmp_path) / "indexes/test_index/metadata.json"
    print(f"\nChecking path: {metadata_path}")
    print(f"Parent exists: {metadata_path.parent.exists()}")
    print(f"Storage path from indexer: {indexer.storage_path}")
    
    assert metadata_path.parent.exists(), "Index directory was not created"
    assert metadata_path.exists(), "Metadata file was not created"
    
    # Verify metadata content
    with open(metadata_path) as f:
        metadata = json.load(f)
        print(f"Loaded metadata: {json.dumps(metadata, indent=2)}")
        assert metadata['index_name'] == 'test_index'
        assert metadata['chunk_size'] == 150
        assert metadata['chunk_overlap'] == 20
        assert metadata['embed_model'] == 'default'
        assert isinstance(metadata['file_hashes'], dict)
        assert isinstance(metadata['last_modified'], dict)

def test_needs_reindexing(tmp_path):
    """Test the needs_reindexing logic"""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Initial content")
    
    options = {
        'index_name': 'test_index',
        'storage_path': str(tmp_path / "indexes/test_index"),
        'chunk_size': 100
    }
    
    indexer = TestIndexer(options)
    
    # First check should indicate need for indexing
    assert indexer.needs_reindexing(str(test_file)) == True
    
    # Update metadata for the file
    indexer.update_file_metadata(str(test_file))
    
    # Should not need reindexing now
    assert indexer.needs_reindexing(str(test_file)) == False
    
    # Modify file content
    time.sleep(0.1)  # Ensure modification time changes
    test_file.write_text("Modified content")
    
    # Should need reindexing after content change
    assert indexer.needs_reindexing(str(test_file)) == True

def test_metadata_persistence(tmp_path):
    """Test that metadata persists between indexer instances"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    storage_path = str(tmp_path / "indexes/test_index")
    print(f"\nStorage path: {storage_path}")
    
    def get_options(chunk_size):
        return {
            'index_name': 'test_index',
            'storage_path': storage_path,
            'chunk_size': chunk_size,
            'chunk_overlap': 10,
            'embed_model': 'test_model'
        }
    
    # First indexer instance
    indexer1 = TestIndexer(get_options(100))
    indexer1.update_file_metadata(str(test_file))
    print(f"Indexer1 metadata: {json.dumps(indexer1.metadata.to_dict(), indent=2)}")
    print(f"Indexer1 chunk_size: {indexer1.chunk_size}")
    
    # Second indexer instance with different parameters
    indexer2 = TestIndexer(get_options(200))
    print(f"Indexer2 metadata: {json.dumps(indexer2.metadata.to_dict(), indent=2)}")
    print(f"Indexer2 chunk_size: {indexer2.chunk_size}")
    
    # Should need reindexing due to different chunk size
    reindex_needed = indexer2.needs_reindexing(str(test_file))
    print(f"Reindex needed with different chunk size: {reindex_needed}")
    assert reindex_needed == True
