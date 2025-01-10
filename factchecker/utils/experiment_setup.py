"""
Utilities for experiment setup and configuration.
"""
import os
import logging
from llama_index.core import Settings

logger = logging.getLogger(__name__)

def configure_logging():
    """Configure basic logging for experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def configure_llama_index():
    """Configure LlamaIndex settings."""
    Settings.chunk_size = 150
    Settings.chunk_overlap = 20

def verify_environment():
    """Verify required environment variables are set."""
    required_vars = ['OPENAI_API_MODEL', 'LLM_TYPE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Environment configuration:")
    logger.info(f"OPENAI_API_MODEL = {os.getenv('OPENAI_API_MODEL')}")
    logger.info(f"LLM_TYPE = {os.getenv('LLM_TYPE')}")

def get_default_indexer_options(source_directory: str = 'data', index_name: str = 'advocate1_index'):
    """Get default indexer options."""
    return [{
        'source_directory': source_directory,
        'index_name': index_name
    }]

def get_default_retriever_options(indexer_options: list, similarity_top_k: int = 8):
    """Get default retriever options."""
    return [{
        'similarity_top_k': similarity_top_k,
        'indexer_options': indexer_options[0]
    }]

def get_default_advocate_options(max_evidences: int = 10, top_k: int = 8, min_score: float = 0.75):
    """Get default advocate options."""
    return {
        'max_evidences': max_evidences,
        'top_k': top_k,
        'min_score': min_score
    } 