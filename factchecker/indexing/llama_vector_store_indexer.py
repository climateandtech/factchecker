"""LlamaVectorStoreIndexer class."""

import logging
<<<<<<< HEAD
from pathlib import Path
import os
import json
from datetime import datetime
=======
from typing import Any, Optional
>>>>>>> 49-sources-subfolder-climatefeedback

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings.utils import EmbedType
<<<<<<< HEAD
from llama_index.core import load_index_from_storage
=======
from llama_index.core.node_parser import SentenceSplitter
>>>>>>> 49-sources-subfolder-climatefeedback

from factchecker.indexing.abstract_indexer import AbstractIndexer
from factchecker.core.embeddings import load_embedding_model


logger = logging.getLogger('factchecker.indexing')

class LlamaVectorStoreIndexer(AbstractIndexer):
    """
    LlamaVectorStoreIndexer class for creating and managing indexes using Llama's VectorStoreIndex.
<<<<<<< HEAD
    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the LlamaVectorStoreIndexer with specified parameters."""
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
=======

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

        # Load embedding model if specified in options
        embedding_kwargs = {}
        if 'embedding_type' in self.options:
            embedding_kwargs['embedding_type'] = self.options.pop('embedding_type')
        if 'embedding_model' in self.options:
            embedding_kwargs['model_name'] = self.options.pop('embedding_model')
        
        self.embed_model = load_embedding_model(**embedding_kwargs)
        self.storage_context_options: dict[str, Any] = self.options.pop('storage_context_options', {})
        self.transformations = self.options.pop('transformations', [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)])
        self.show_progress = self.options.pop('show_progress', True)

>>>>>>> 49-sources-subfolder-climatefeedback

    def build_index(self, documents: list[Document]) -> None:
        """
        Builds a new VectorStoreIndex from the provided documents.
        
        Args:
<<<<<<< HEAD
            documents (List[Document]): Documents to index
=======
            documents (list[Document]): list of LlamaIndex Documents to index.

        Raises:
            Exception: If an error occurs during index creation.

>>>>>>> 49-sources-subfolder-climatefeedback
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

<<<<<<< HEAD
    def add_to_index(self, documents: List[Any]) -> None:
        """Add documents to the index."""
        if self.index is None:
            raise ValueError("Index not initialized")
        self.index.insert_nodes(documents)
        self.save_index()

    def delete_from_index(self, document_ids: List[str]) -> None:
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
=======
        Raises:
            NotImplementedError: If the method is not yet implemented.

        """
        logging.error("load_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("load_index() of LlamaVectorStoreIndexer is not yet implemented")

    def add_to_index(self, documents: list[Document]) -> None:
        """
        Add documents to the index.

        Args:
            documents (list[Document]): Documents to be added to the index.

        Raises:
            NotImplementedError: If the method is not yet implemented.

        """
        logging.error("add_to_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of LlamaVectorStoreIndexer is not yet implemented")

    def delete_from_index(self, document_ids: list[str]) -> None:
        """
        Delete documents from the index.

        Args:
            document_ids (list[str]): list of document IDs to delete from the index.
>>>>>>> 49-sources-subfolder-climatefeedback

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
