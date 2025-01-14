"""
Module for creating and configuring query engines.

This module provides functionality to create and configure query engines
for use with document collections and various embedding models.
"""

from llama_index.core import VectorStoreIndex, Settings

def create_query_engine(documents, embed_model, llm):
    """
    Create a query engine with the specified documents and models.
    
    Args:
        documents (list[Document]): List of documents to index.
        embed_model (BaseEmbedding): Embedding model to use for document vectorization.
        llm (BaseLLM): Language model to use for query processing.
        
    Returns:
        QueryEngine: Configured query engine ready for use.
    """
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(documents)
    
    # Return configured query engine
    return index.as_query_engine() 