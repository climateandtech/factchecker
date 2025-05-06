"""Ollama BGE retriever implementation with sparse filtering and hybrid search."""

from typing import List, Dict, Any, Optional
import os
import requests
import json
import logging

from factchecker.retrieval.abstract_retriever import AbstractRetriever
from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.core.embeddings import load_embedding_model
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

class OllamaBGERetriever(AbstractRetriever):
    """Retriever that uses Ollama's BGE-M3 model for sparse filtering and hybrid retrieval."""
    
    def __init__(
        self,
        indexer: LlamaVectorStoreIndexer,
        options: Dict[str, Any]
    ):
        """Initialize the retriever.
        
        Args:
            indexer: The indexer to use for document storage and retrieval
            options: Configuration options including:
                - top_k: Number of documents to retrieve
                - rerank_top_k: Number of documents to consider for reranking
                - base_url: Base URL for the Ollama API
                - similarity_cutoff: Minimum similarity score for documents
        """
        super().__init__(indexer)
        self.top_k = options.get('top_k', 5)
        self.rerank_top_k = options.get('rerank_top_k', 20)
        self.base_url = options.get('base_url', 'http://localhost:11434')
        self.similarity_cutoff = options.get('similarity_cutoff', 0.0)
        
        # Create retriever
        self.create_retriever()
    
    def create_retriever(self):
        """Create the retriever using the BGE model."""
        # Initialize the index if not already initialized
        if self.indexer.index is None:
            self.indexer.initialize_index()
        
        # Create the retriever
        self.retriever = self.indexer.index.as_retriever(
            similarity_top_k=self.rerank_top_k
        )
    
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Retrieve documents using BGE embeddings.
        
        Args:
            query: The query to retrieve documents for
            filters: Optional filters to apply to the retrieval
            top_k: Optional override for number of documents to retrieve
            
        Returns:
            List of NodeWithScore objects containing retrieved documents and their scores
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Use larger top_k for initial retrieval
        initial_top_k = top_k or self.rerank_top_k
        logger.info(f"Using initial top_k: {initial_top_k}")
        
        # Get initial candidates using the retriever
        logger.info("Getting initial candidates from retriever...")
        if self.retriever is None:
            self.create_retriever()
            
        nodes = self.retriever.retrieve(query)
        
        # Log raw nodes and scores
        logger.info("Raw nodes from retriever:")
        for i, node in enumerate(nodes):
            if hasattr(node, 'score'):
                logger.info(f"Node {i}: score={node.score:.3f}")
                logger.info(f"Text preview: {node.node.text[:200]}...")
            else:
                logger.info(f"Node {i}: no score available")
                logger.info(f"Text preview: {node.node.text[:200]}...")
        
        if not nodes:
            logger.warning("Initial retrieval returned no candidates")
            return []
            
        candidates = []
        for node in nodes:
            if hasattr(node, 'score') and node.score >= self.similarity_cutoff:
                candidates.append(node)
            
        logger.info(f"After filtering (cutoff={self.similarity_cutoff}): {len(candidates)} candidates")
        if candidates:
            logger.info("First filtered candidate preview:")
            logger.info(f"{candidates[0].node.text[:200]}...")
        
        return candidates[:top_k or self.top_k] 