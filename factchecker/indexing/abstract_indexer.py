"""Abstract base class for creating and managing indexes."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from pathlib import Path
import json
import hashlib
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from factchecker.utils.logging_config import setup_logging

logger = logging.getLogger('factchecker.indexing')

class IndexMetadata:
    def __init__(self, index_name: str, chunk_size: int, chunk_overlap: int, embed_model: str):
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = embed_model
        self.file_hashes: Dict[str, str] = {}
        self.last_modified: Dict[str, float] = {}
        self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_name": self.index_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embed_model": self.embed_model,
            "file_hashes": self.file_hashes,
            "last_modified": self.last_modified,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        metadata = cls(
            data["index_name"],
            data["chunk_size"],
            data["chunk_overlap"],
            data["embed_model"]
        )
        metadata.file_hashes = data["file_hashes"]
        metadata.last_modified = data["last_modified"]
        metadata.created_at = data["created_at"]
        metadata.last_updated = data["last_updated"]
        return metadata

def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class AbstractIndexer(ABC):
    """
    Abstract base class for creating and managing indexes.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        source_directory (str): Directory containing source data files.
        initial_documents (Optional[list[Document]]): Initial documents to be indexed.
        initial_files (Optional[list[str]]): Initial files to be indexed.
        index (Optional[Any]): In-memory index object.

    """

    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the AbstractIndexer with configuration options.

        Args:
            options (Optional[Dict[str, Any]]): Dictionary containing configuration options. 
                If not provided, defaults will be used.

        """
        logger.debug(f"Initializing indexer with options: {options}")
        self.options = options or {}
        self.index_name = self.options.pop('index_name', 'default_index')
        self.initial_documents = self.options.pop('documents', None)
        self.storage_path = Path(self.options.pop('storage_path', f"indexes/{self.index_name}"))
        self.chunk_size = self.options.pop('chunk_size', 150)
        self.chunk_overlap = self.options.pop('chunk_overlap', 20)
        self.embed_model = self.options.pop('embed_model', 'default')
        self.index: Optional[Any] = None
        self.metadata = self._load_or_create_metadata()
        self._parameters_changed = self._check_parameters_changed()  # Cache this result
        self._save_metadata()

    def _get_metadata_path(self) -> Path:
        """Get the path for the metadata file"""
        return self.storage_path / "metadata.json"

    def _load_or_create_metadata(self) -> IndexMetadata:
        """Load existing metadata or create new if not exists"""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    metadata = IndexMetadata.from_dict(data)
                    # Use the storage path from metadata
                    self.storage_path = Path(data.get("storage_path", self.storage_path))
                    return metadata
            except Exception as e:
                logging.error(f"Error loading metadata from {metadata_path}: {e}")
        
        # Create new metadata if file doesn't exist or there was an error
        return IndexMetadata(
            self.index_name,
            self.chunk_size,
            self.chunk_overlap,
            self.embed_model
        )

    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        metadata_dict = self.metadata.to_dict()
        metadata_dict["storage_path"] = str(self.storage_path)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self._get_metadata_path(), 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def _check_parameters_changed(self) -> bool:
        """Check if indexing parameters have changed once during initialization"""
        if not hasattr(self, 'metadata'):
            return True
        return (self.chunk_size != self.metadata.chunk_size or
                self.chunk_overlap != self.metadata.chunk_overlap or
                self.embed_model != self.metadata.embed_model)

    def needs_reindexing(self, file_path: str) -> bool:
        """Check if a file needs to be reindexed"""
        if not Path(file_path).exists():
            logger.debug(f"File {file_path} does not exist")
            return True

        # Use cached parameter check result
        if self._parameters_changed:
            logger.info("🔄 Reindexing needed: parameters changed")
            return True

        # Only check file changes if parameters haven't changed
        current_hash = calculate_file_hash(file_path)
        last_modified = Path(file_path).stat().st_mtime

        if file_path in self.metadata.file_hashes:
            stored_hash = self.metadata.file_hashes[file_path]
            stored_modified = self.metadata.last_modified[file_path]
            
            if current_hash != stored_hash or last_modified != stored_modified:
                logger.info(f"🔄 Reindexing needed: file {file_path} changed")
                return True
            logger.debug(f"✅ File {file_path} unchanged")
            return False

        logger.debug(f"🆕 New file {file_path}")
        return True

    def update_file_metadata(self, file_path: str) -> None:
        """Update metadata for a processed file"""
        self.metadata.file_hashes[file_path] = calculate_file_hash(file_path)
        self.metadata.last_modified[file_path] = Path(file_path).stat().st_mtime
        self._save_metadata()

    def save_index(self) -> None:
        """Save the index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save")
            
        # Save LlamaIndex data
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(self.storage_path))
        
        # Save our metadata
        self._save_metadata()
        logging.info(f"Index and metadata saved to {self.storage_path}")

    def load_index(self) -> None:
        """Load the index from disk"""
        if not self.storage_path.exists():
            raise FileNotFoundError(f"No index found at {self.storage_path}")
            
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path)
            )
            self.index = load_index_from_storage(storage_context)
            logging.info(f"Index loaded from {self.storage_path}")
        except Exception as e:
            logging.error(f"Error loading index: {e}")
            raise

    def load_initial_documents(self) -> List[Document]:
        """Load documents from various sources, with support for caching and reindexing."""
        try:
            # Use preloaded documents if provided
            if hasattr(self, 'initial_documents') and self.initial_documents:
                logger.info("📄 Using preloaded documents provided in options.")
                return self.initial_documents

            # Use specified files if provided
            if hasattr(self, 'initial_files') and self.initial_files:
                files = self.initial_files
                logger.info(f"📄 Loading documents from specified files: {files}")
                documents = SimpleDirectoryReader(input_files=files).load_data()
                self.initial_documents = documents
                logger.debug(f"Loaded {len(documents)} documents from specified files")
                return documents

            # Default behavior: load from source directory with caching and change detection
            source_dir = Path(self.options.get('source_directory', 'data'))
            logger.debug(f"Checking documents in {source_dir}")
            needs_parsing = False

            # Check parameter changes
            if (self.chunk_size != self.metadata.chunk_size or
                self.chunk_overlap != self.metadata.chunk_overlap):
                logger.info(f"🔄 Parameters changed:")
                logger.info(f"  - chunk_size: {self.chunk_size} != {self.metadata.chunk_size}")
                logger.info(f"  - chunk_overlap: {self.chunk_overlap} != {self.metadata.chunk_overlap}")
                needs_parsing = True

            # Check file changes
            if not needs_parsing:
                for file_path in source_dir.glob('*'):
                    if self.needs_reindexing(str(file_path)):
                        logger.info(f"🔄 File {file_path} needs parsing")
                        needs_parsing = True
                        break

            # Try loading cached documents
            if not needs_parsing and self.storage_path.exists():
                try:
                    docs_cache_path = self.storage_path / "documents.json"
                    if docs_cache_path.exists():
                        logger.info("📄 Loading cached documents (fast path)")
                        with open(docs_cache_path, "r") as f:
                            docs_data = json.load(f)
                            documents = [Document.from_dict(d) for d in docs_data]
                            self.initial_documents = documents
                            return documents
                    else:
                        logger.info("⚠️ No cached documents found")
                        needs_parsing = True
                except Exception as e:
                    logger.warning(f"⚠️ Could not load cached documents: {e}")
                    needs_parsing = True

            # Parse and cache documents if needed
            if needs_parsing:
                logger.info("📚 Parsing documents from source (slow path)...")
                reader = SimpleDirectoryReader(source_dir)
                documents = reader.load_data()
                self.storage_path.mkdir(parents=True, exist_ok=True)
                with open(self.storage_path / "documents.json", "w") as f:
                    json.dump([doc.to_dict() for doc in documents], f)
                logger.info(f"✅ Parsed and cached {len(documents)} documents")
                self.initial_documents = documents
                return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def check_persisted_index_exists(self) -> bool:
        """
        Determine if a persisted index exists at the specified path.

        Returns:
            bool: True if the persisted index exists, False otherwise.
            
        """
        if self.index_path and os.path.exists(self.index_path):
            logging.debug(f"Index exists at {self.index_path}")
            return True
        logging.debug("No persisted index found.")
        return False

    def initialize_index(self) -> None:
        """
        Initializes the index by loading an existing index or building a new one.
        
        This includes:
        - Checking if an in-memory index exists
        - Checking if a stored index exists and can be reused
        - Detecting if reindexing is required (via param or file changes)
        - Building a new index if needed
        """
        try:
            if self.index is not None:
                logger.info("✅ In-memory index exists, skipping initialization")
                return

            logger.info("🔍 Checking if an index already exists on disk...")
            if self.storage_path.exists():
                logger.info(f"📂 Found existing index at {self.storage_path}")
                source_dir = Path(self.options.get('source_directory', 'data'))
                needs_reindex = False

                for file_path in source_dir.glob('*'):
                    if self.needs_reindexing(str(file_path)):
                        needs_reindex = True
                        break

                if not needs_reindex:
                    try:
                        logger.info("🔄 Loading existing index from disk...")
                        self.load_index()
                        logger.info("✅ Successfully loaded cached index")
                        return
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load existing index: {e}")
                        # Fall through to rebuild index

            # No valid cached index or reindexing needed
            logger.info("🏗️ Building new index from scratch...")
            documents = self.load_initial_documents()
            logger.info(f"📄 Loaded {len(documents)} documents")

            self.build_index(documents)

            # Update metadata for all processed files
            source_dir = Path(self.options.get('source_directory', 'data'))
            for file_path in source_dir.glob('*'):
                self.update_file_metadata(str(file_path))

            self.save_index()
            logger.info("✅ Index built and saved successfully")

        except FileNotFoundError as e:
            logger.error(f"❌ File not found during index initialization: {e}")
            raise
        except ValueError as e:
            logger.error(f"❌ Invalid data during index initialization: {e}")
            raise
        except Exception as e:
            logger.exception(f"❌ Error during index initialization: {e}")
            raise

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """
        Builds the index from the provided documents.

        Args:
            documents (List[Any]): The documents to index.

        """
        pass

    @abstractmethod
    def add_to_index(self, documents: List[Any]) -> None:
        """
        Add documents to the index.

        Args:
            documents (List[Any]): Documents to be added to the index.

        """
        pass

    @abstractmethod
    def delete_from_index(self, document_ids: List[Any]) -> None:
        """
        Delete documents from the index based on their IDs.

        Args:
            document_ids (List[Any]): IDs of documents to be removed from the index.

        """
        pass
