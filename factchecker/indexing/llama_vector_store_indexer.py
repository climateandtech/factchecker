"""LlamaVectorStoreIndexer class."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.node_parser import SentenceSplitter

from factchecker.core.embeddings import load_embedding_model
from factchecker.indexing.abstract_indexer import AbstractIndexer

logger = logging.getLogger('factchecker.indexing')

class LlamaVectorStoreIndexer(AbstractIndexer):
    """
    LlamaVectorStoreIndexer class for creating and managing indexes using Llama's VectorStoreIndex.

    Attributes:
        options (dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        index (Optional[Any]): In-memory index object.
        embed_model (Optional[EmbedType]): The name of the embedding model to use.
        storage_context_options (dict[str, Any]): Options for the storage context.
        transformations (list[Callable]): A list of transformations to apply to the documents.
        show_progress (bool): Whether to show progress during indexing.
    """

    def __init__(self, options: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the LlamaVectorStoreIndexer with specified parameters.
        
        Args:
        options (Optional[dict[str, Any]]): Configuration options which may include:
            - index_name (str): Name of the index. Defaults to 'default_index'.
            - index_path (Optional[str]): Path to the directory where the index is stored on disk.
            - source_directory (str): Directory containing source data files. Defaults to 'data'.
            - storage_context_options (dict[str, Any]): Options for the storage context.
            - transformations (list[Callable]): A list of transformations to apply to the documents.
            - embedding_type (str): Type of embedding model to use.
            - embedding_model (str): Name of the embedding model to use.
            - storage_context_options (Dict[str, Any]): Options for the storage context.
            - transformations (List[Callable]): A list of transformations to apply to the documents.
            - show_progress (bool): Whether to show progress during indexing.
        """
        super().__init__(options)
        
        # Just set the path, don't create yet
        base_dir = Path.cwd() / "storage"
        self.storage_path = base_dir / "indices" / self.index_name
        
        logger.debug(f"Using storage path: {self.storage_path}")
        
        self.storage_context = None
        self.transformations = [
            SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        ]
        self.show_progress = self.options.pop('show_progress', False)

    def build_index(self, documents: list[Document]) -> None:
        """
        Builds a new VectorStoreIndex from the provided documents.
        
        Args:
            documents (List[Document]): Documents to index
        """
        try:
            logger.info(f"Building VectorStoreIndex with {len(documents)} documents")
            
            # Create empty storage context first
            self.storage_context = StorageContext.from_defaults()
            
            # Build index in memory with storage context
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                transformations=self.transformations,
                show_progress=True
            )
            
            # Then persist to disk
            self.storage_context.persist(persist_dir=str(self.storage_path))
            
            logger.info("Index built and persisted successfully")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise

    def save_index(self) -> None:
        """Save the LlamaIndex vector store index to disk."""
        try:
            if self.index is None:
                raise ValueError("No index to save - index must be built first")
            
            # Ensure storage directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create storage context if needed
            if self.storage_context is None:
                self.storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.storage_path)
                )
            
            # Save using existing context
            self.storage_context.persist()  # No need to specify persist_dir again
            logger.info(f"Index saved successfully to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

    def load_index(self) -> None:
        """Load the LlamaVectorStore index from disk."""
        try:
            # Check for both LlamaIndex files and our metadata
            required_files = [
                self.storage_path / "docstore.json",
                self.storage_path / "metadata.json"
            ]
            
            if not all(f.exists() for f in required_files):
                raise FileNotFoundError(
                    f"Missing required files in {self.storage_path}"
                )
            
            # Load our metadata first
            with open(self.storage_path / "metadata.json") as f:
                metadata = json.load(f)
                # Verify metadata matches current configuration
                if metadata.get('chunk_size') != self.chunk_size:
                    raise ValueError("Index was built with different chunk_size")
            
            # Then load LlamaIndex files
            self.storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path)
            )
            
            self.index = load_index_from_storage(
                storage_context=self.storage_context
            )
            logger.info(f"Index loaded successfully from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def add_to_index(self, documents: list[Any]) -> None:
        """Add documents to the index."""
        if self.index is None:
            raise ValueError("Index not initialized")
        self.index.insert_nodes(documents)
        self.save_index()

    def delete_from_index(self, document_ids: list[str]) -> None:
        """Delete documents from the index based on their IDs."""
        if self.index is None:
            raise ValueError("Index not initialized")
        self.index.delete_nodes(document_ids)
        self.save_index()

    def initialize_index(self) -> None:
        """Initialize or load the index."""
        try:
            # First check if we have an existing valid index
            if self.storage_path.exists():
                try:
                    logger.info("🔍 Found existing index, attempting to load...")
                    self.load_index()
                    logger.info("✅ Successfully loaded cached index")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Could not load existing index: {e}")
                    logger.info("🔄 Will rebuild index...")

            # Create new index
            logger.info("🏗️ Building new index...")
            documents = self.load_initial_documents()
            logger.info(f"📄 Loaded {len(documents)} documents")
            
            # Build the index
            self.build_index(documents)
            
            # Save both the index and our metadata
            self.save_index()
            self.update_metadata()  # This saves our metadata.json
            
            logger.info("✅ Index built and saved successfully")

        except Exception as e:
            logger.exception(f"❌ Error during index initialization: {e}")
            raise

    def update_metadata(self) -> None:
        """Update and save metadata after index operations."""
        # Update metadata with current configuration
        self.metadata.last_updated = datetime.now().isoformat()
        
        # For directory-based indexing, update file metadata
        if 'source_directory' in self.options:
            source_dir = Path(self.options['source_directory'])
            for file_path in source_dir.glob('*'):
                self.update_file_metadata(str(file_path))
        
        # Save metadata
        metadata_path = self.storage_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        logger.debug(f"Updated metadata at {metadata_path}")
