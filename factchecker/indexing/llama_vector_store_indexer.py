"""LlamaVectorStoreIndexer class."""

import logging
import os
from typing import Any, Optional

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from factchecker.indexing.abstract_indexer import AbstractIndexer
from factchecker.core.embeddings import load_embedding_model

logger = logging.getLogger(__name__)


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


    def build_index(self, documents: list[Document]) -> None:
        """
        Build the LlamaVectorStore index from the provided documents.

        Args:
            documents (list[Document]): list of LlamaIndex Documents to index.

        Raises:
            Exception: If an error occurs during index creation.

        """
        try:
            logger.info("Building vector index (chunking and embedding documents)...")
            storage_context = StorageContext.from_defaults(**self.storage_context_options)
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=self.transformations,
                show_progress=self.show_progress,
            )
            logger.info("VectorStoreIndex successfully built")
        
        except Exception as e:
            logger.exception(f"Failed to create LlamaVectorStore index: {e}")
            raise

    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Save the LlamaVectorStore index to disk.

        Args:
            index_path (Optional[str]): Path where the index will be saved.
                Defaults to self.index_path. Required if self.index_path is not set.

        Raises:
            ValueError: If no index_path is available and none is provided.
        """
        persist_dir = index_path or self.index_path
        if not persist_dir:
            raise ValueError("index_path is required to save the index")
        if self.index is None:
            raise ValueError("No index to save; build the index first")
        os.makedirs(persist_dir, exist_ok=True)
        self.index.storage_context.persist(persist_dir=persist_dir)
        logger.info("VectorStoreIndex saved to %s", persist_dir)

    def load_index(self) -> None:
        """
        Load the LlamaVectorStore index from disk.

        Raises:
            ValueError: If index_path is not set or no persisted index exists.
        """
        if not self.index_path:
            raise ValueError("index_path is required to load the index")
        if not os.path.isdir(self.index_path):
            raise ValueError(f"No persisted index found at {self.index_path}")
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(
            storage_context, embed_model=self.embed_model
        )
        logger.info("VectorStoreIndex loaded from %s", self.index_path)

    def add_to_index(self, documents: list[Document]) -> None:
        """
        Add documents to the index.

        Args:
            documents (list[Document]): Documents to be added to the index.

        Raises:
            NotImplementedError: If the method is not yet implemented.

        """
        logger.error("add_to_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of LlamaVectorStoreIndexer is not yet implemented")

    def delete_from_index(self, document_ids: list[str]) -> None:
        """
        Delete documents from the index.

        Args:
            document_ids (list[str]): list of document IDs to delete from the index.

        Raises:    
            NotImplementedError: If the method is not yet implemented.

        """
        logger.error("delete_from_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of LlamaVectorStoreIndexer is not yet implemented")
