"""LlamaVectorStoreIndexer class."""

import logging
import logging.handlers
from typing import Any, Optional, Callable, List, Iterator, Dict
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
import uuid

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TextNode

from factchecker.indexing.abstract_indexer import AbstractIndexer
from factchecker.core.embeddings import DirectOllamaEmbedding, load_embedding_model

logger = logging.getLogger(__name__)

# Create a wrapper class to make our embeddings compatible with llama-index
class LlamaIndexEmbeddingWrapper(BaseEmbedding):
    def __init__(self, embed_model: DirectOllamaEmbedding):
        super().__init__()
        self._embed_model = embed_model
        
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_model.get_text_embedding(query)
        
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_model.get_text_embedding(text)
        
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_model.get_text_embeddings(texts)
        
    async def _aget_query_embedding(self, query: str) -> List[float]:
        # For now, just call the sync version since our underlying model is sync
        return self._get_query_embedding(query)

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
                - batch_size (int): Size of batches when processing texts. Defaults to 100.
        """
        super().__init__(options)

        # Load embedding model if specified in options
        embedding_kwargs = {}
        if 'embedding_type' in self.options:
            embedding_kwargs['embedding_type'] = self.options.pop('embedding_type')
        if 'embedding_model' in self.options:
            embedding_kwargs['model_name'] = self.options.pop('embedding_model')
        
        # Load our embeddings and wrap them for llama-index
        direct_embeddings = load_embedding_model(**embedding_kwargs)
        self.embed_model = LlamaIndexEmbeddingWrapper(direct_embeddings)
        
        self.storage_context_options = self.options.pop('storage_context_options', {})
        self.node_batch_size = self.options.pop('node_batch_size', 1000)
        self.embedding_batch_size = self.options.pop('embedding_batch_size', 64)
        self.num_workers = self.options.pop('num_workers', 6)
        self.show_progress = self.options.pop('show_progress', True)
        
        # Get chunk size and overlap from options
        self.chunk_size = self.options.pop('chunk_size', 150)
        self.chunk_overlap = self.options.pop('chunk_overlap', 20)
        
        # Set up logging for parallel processing
        self._setup_logging()
        
        # Configure node parser with settings from options
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Create ingestion pipeline with transformations
        self.pipeline = IngestionPipeline(
            transformations=[
                self.node_parser,  # Split text into chunks
            ]
        )

        # Ensure we have a valid index path
        if not self.index_path:
            if self.source_directory:
                self.index_path = os.path.join(self.source_directory, self.index_name)
            else:
                self.index_path = os.path.join("data", "indices", self.index_name)
        os.makedirs(self.index_path, exist_ok=True)

    def _setup_logging(self):
        """Set up queue-based logging for worker processes."""
        self.log_queue = multiprocessing.Queue()
        self.queue_handler = logging.handlers.QueueHandler(self.log_queue)
        
        # Create and start queue listener
        self.queue_listener = logging.handlers.QueueListener(
            self.log_queue,
            *logging.getLogger().handlers
        )
        self.queue_listener.start()

    def _ensure_unique_node_ids(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Ensure all nodes have unique IDs to prevent KeyError during embedding."""
        # Check for duplicate node IDs
        node_ids = [node.id_ for node in nodes]
        unique_ids = set(node_ids)
        
        if len(unique_ids) != len(nodes):
            logger.warning(f"Found {len(nodes) - len(unique_ids)} duplicate node IDs - assigning new IDs")
            
            # Create a map of ID occurrences
            id_occurrences = {}
            for id in node_ids:
                id_occurrences[id] = id_occurrences.get(id, 0) + 1
            
            # Assign new IDs to duplicates
            for i, node in enumerate(nodes):
                if id_occurrences[node.id_] > 1:
                    # Generate a new unique ID
                    new_id = str(uuid.uuid4())
                    logger.debug(f"Changing duplicate node ID {node.id_} to {new_id}")
                    node.id_ = new_id
                    id_occurrences[node.id_] -= 1  # Decrement count for the original ID
        
        return nodes

    def build_index(self, documents: List[Document]) -> None:
        """
        Build the index with all documents in one step.
        This simpler approach avoids the node ID issues that can occur with batch processing.
        """
        try:
            logger.info(f"Building index with {len(documents)} documents")
            
            # Process all documents at once
            logger.info("Creating nodes from documents...")
            nodes = self.pipeline.run(documents=documents)
            logger.info(f"Created {len(nodes)} nodes from documents")
            
            # Build the index directly
            logger.info("Building vector store index...")
            storage_context = StorageContext.from_defaults(**self.storage_context_options)
            
            # Initialize Settings with our embed model
            Settings.embed_model = self.embed_model
            logger.info(f"Using embed model: {type(self.embed_model).__name__}")
            logger.info(f"Batch size for embeddings: {self.embedding_batch_size}")
            
            # Build the index with all nodes at once
            logger.info(f"Building index with {len(nodes)} nodes...")
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                show_progress=self.show_progress,
                embed_model=self.embed_model,
            )
            
            logger.info("Index built successfully")
            
        except Exception as e:
            logger.exception(f"Failed to build index: {e}")
            raise

    def initialize_index(self) -> None:
        """Initialize an existing index."""
        try:
            storage_context = StorageContext.from_defaults(**self.storage_context_options)
            self.index = VectorStoreIndex(
                nodes=[],  # Empty nodes since we're loading from storage
                storage_context=storage_context,
                show_progress=self.show_progress,
                embed_model=self.embed_model
            )
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load index: {e}")
            raise

    def save_index(self) -> None:
        """Save the LlamaVectorStore index to disk."""
        if not self.index:
            logger.error("No index to save")
            return
            
        try:
            logger.info(f"Saving index to {self.index_path}")
            self.index.storage_context.persist(persist_dir=self.index_path)
            logger.info("Index saved successfully")
        except Exception as e:
            logger.exception(f"Failed to save index: {e}")
            raise

    def load_index(self) -> None:
        """Load the LlamaVectorStore index from disk."""
        try:
            if not os.path.exists(self.index_path):
                logger.error(f"Index path {self.index_path} does not exist")
                return

            logger.info(f"Loading index from {self.index_path}")
            storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                show_progress=self.show_progress,
                embed_model=self.embed_model
            )
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load index: {e}")
            raise

    def add_to_index(self, documents: list[Document]) -> None:
        """Add documents to the index."""
        try:
            if not self.index:
                logger.error("No index to add to")
                return
                
            logger.info(f"Adding {len(documents)} documents to index")
            
            # Create nodes
            nodes = self.pipeline.run(documents=documents)
            nodes = self._ensure_unique_node_ids(nodes)
            
            # Add to index
            self.index.insert_nodes(nodes=nodes)
            logger.info(f"Added {len(nodes)} nodes to index")
        except Exception as e:
            logger.exception(f"Failed to add documents to index: {e}")
            raise

    def delete_from_index(self, document_ids: list[str]) -> None:
        """Delete documents from the index."""
        try:
            if not self.index:
                logger.error("No index to delete from")
                return
                
            logger.info(f"Deleting {len(document_ids)} documents from index")
            for doc_id in document_ids:
                self.index.delete(doc_id)
            logger.info(f"Deleted {len(document_ids)} documents from index")
        except Exception as e:
            logger.exception(f"Failed to delete documents from index: {e}")
            raise

    def __del__(self):
        """Clean up resources when the indexer is destroyed."""
        if hasattr(self, 'queue_listener') and self.queue_listener:
            self.queue_listener.stop()
            self.log_queue.close()
