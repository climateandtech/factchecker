"""This module contains the LlamaVectorStoreIndexer class."""

from typing import Any, Dict, List, Optional, Callable
import logging

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings.utils import EmbedType

from factchecker.indexing.abstract_indexer import AbstractIndexer
from factchecker.core.embeddings import load_embedding_model

class LlamaVectorStoreIndexer(AbstractIndexer):
    """
    LlamaVectorStoreIndexer class for creating and managing indexes using Llama's VectorStoreIndex.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        index (Optional[Any]): In-memory index object.
        ---
        embed_model (Optional[EmbedType]): The name of the embedding model to use.
        storage_context_options (Dict[str, Any]): Options for the storage context.
        transformations (List[Callable]): A list of transformations to apply to the documents.
        show_progress (bool): Whether to show progress during indexing.

    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the LlamaVectorStoreIndexer with specified parameters.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options which may include:
                - index_name (str): Name of the index. Defaults to 'default_index'.
                - index_path (Optional[str]): Path to the directory where the index is stored on disk.
                - source_directory (str): Directory containing source data files. Defaults to 'data'.
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
        self.storage_context_options = self.options.get('storage_context_options', {})
        self.transformations = self.options.get('transformations', [])
        self.show_progress = self.options.get('show_progress', True)


    def build_index(self, documents: List[Document]) -> None:
        """
        Build the LlamaVectorStore index from the provided documents.

        Args:
            documents (List[Document]): List of LlamaIndex Documents to index.

        Raises:
            Exception: If an error occurs during index creation.

        """
        try:

            storage_context = StorageContext.from_defaults(**self.storage_context_options)
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=self.transformations,
                show_progress=self.show_progress,
            )
            logging.info("VectorStoreIndex successfully built")
        
        except Exception as e:
            logging.exception(f"Failed to create LlamaVectorStore index: {e}")
            raise

    def save_index(self) -> None:
        """
        Save the LlamaVectorStore index to disk.

        Raises:
            NotImplementedError: If the method is not yet implemented.

        """
        logging.error("save_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("save_index() of LlamaVectorStoreIndexer is not yet implemented")

    def load_index(self) -> None:
        """
        Load the LlamaVectorStore index from disk.

        Raises:
            NotImplementedError: If the method is not yet implemented.

        """
        logging.error("load_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("load_index() of LlamaVectorStoreIndexer is not yet implemented")

    def add_to_index(self, documents: List[Document]) -> None:
        """
        Add documents to the index.

        Args:
            documents (List[Document]): Documents to be added to the index.

        Raises:
            NotImplementedError: If the method is not yet implemented.

        """
        logging.error("add_to_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("add_to_index() of LlamaVectorStoreIndexer is not yet implemented")

    def delete_from_index(self, document_ids: List[str]) -> None:
        """
        Delete documents from the index.

        Args:
            document_ids (List[str]): List of document IDs to delete from the index.

        Raises:    
            NotImplementedError: If the method is not yet implemented.

        """
        logging.error("delete_from_index() of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index() of LlamaVectorStoreIndexer is not yet implemented")
