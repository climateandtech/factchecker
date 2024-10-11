"""This module contains the LlamaVectorStoreIndexer class, which is a subclass of AbstractIndexer."""

from factchecker.indexing.abstract_indexer import AbstractIndexer
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings.utils import EmbedType
from typing import Any, Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class LlamaVectorStoreIndexer(AbstractIndexer):
    """
    LlamaVectorStoreIndexer class for creating and managing indexes using Llama's VectorStoreIndex.

    Attributes:
        options (Dict[str, Any]): Configuration options for the indexer.
        index_name (str): Name of the index.
        index_path (Optional[str]): Path to the directory where the index is stored on disk.
        index (Optional[Any]): In-memory index object.
        embed_model (Optional[EmbedType]): The name of the embedding model to use.
        vector_store (Optional[str]): The name of the vector store to use.
        transformations (List[Callable]): A list of transformations to apply to the documents.
        show_progress (bool): Whether to show progress during indexing.
    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(options)
        self.embed_model: Optional[EmbedType] = self.options.pop('embed_model', None)
        self.vector_store: Optional[str] = self.options.pop('vector_store', None)
        self.transformations: List[Callable] = self.options.pop(
            'transformations', 
            [SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)]
        )
        self.show_progress: bool = self.options.pop('show_progress', False)


    def build_index(self, documents: List[Document]) -> None:
        try:

            # Now self.options should only contain relevant options for StorageContext.from_defaults
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store, **self.options)
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            logger.info("VectorStoreIndex successfully built")
        
        except Exception as e:
            logger.exception(f"Failed to create LlamaVectorStore index: {e}")
            raise

    def save_index(self):
        logger.error("save_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("save_index of LlamaVectorStoreIndexer is not yet implemented")

    def load_index(self):
        logger.error("load_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("load_index of LlamaVectorStoreIndexer is not yet implemented")

    def add_to_index(self, documents):
        logger.error("add_to_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("add_to_index of LlamaVectorStoreIndexer is not yet implemented")

    def delete_from_index(self, document_ids):
        logger.error("delete_from_index of LlamaVectorStoreIndexer is not yet implemented")
        raise NotImplementedError("delete_from_index of LlamaVectorStoreIndexer is not yet implemented")
