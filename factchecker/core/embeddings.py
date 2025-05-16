import os
from typing import Optional, List, Dict, Any
import logging
import logging.handlers
from ollama import Client as OllamaClient
import numpy as np

logger = logging.getLogger(__name__)

# Set up a dedicated file logger for embedding progress
embedding_logger = logging.getLogger("embedding_progress")
embedding_logger.setLevel(logging.INFO)
embedding_logger.propagate = False  # Don't send to parent logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Check if handler already exists to avoid duplicates
if not embedding_logger.handlers:
    # Create a file handler for the embedding logger
    embedding_file_handler = logging.FileHandler("logs/embedding_progress.log", mode="w")
    embedding_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    embedding_logger.addHandler(embedding_file_handler)
    
    # Also add a console handler for direct visibility
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('EMBEDDING: %(message)s'))
    embedding_logger.addHandler(console_handler)

# Log startup message to confirm logger is working
embedding_logger.info("Embedding progress logger initialized")

class DirectOllamaEmbedding:
    """Direct Ollama embedding class using the official client."""
    
    def __init__(self, model_name: str = "bge-m3", base_url: Optional[str] = None):
        """Initialize with direct Ollama client."""
        self._client = OllamaClient(host=base_url) if base_url else OllamaClient()
        self._model = model_name
        self._embedding_cache: Dict[str, List[float]] = {}
        embedding_logger.info(f"DirectOllamaEmbedding initialized with model {model_name}")
        
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embedding_logger.info("ENTRY: get_text_embedding called for single text")
        # Check cache first
        if text in self._embedding_cache:
            embedding_logger.info("Cache hit for single text embedding")
            return self._embedding_cache[text]
            
        embedding_logger.info(f"Getting single embedding for text of length {len(text)}")
        response = self._client.embeddings(model=self._model, prompt=text)
        embedding_logger.info("Successfully got single embedding")
        embedding = response['embedding']
        self._embedding_cache[text] = embedding
        return embedding

    def get_text_embeddings(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embedding_logger.info(f"ENTRY: get_text_embeddings called with {len(texts)} texts")
        # Check which texts need to be embedded
        texts_to_embed = [text for text in texts if text not in self._embedding_cache]
        
        if texts_to_embed:
            # Process each text individually since Ollama doesn't support true batching
            embedding_logger.info(f"Getting embeddings for {len(texts_to_embed)} texts")
            for i, text in enumerate(texts_to_embed):
                if show_progress and i % 10 == 0:
                    embedding_logger.info(f"Processing text {i+1}/{len(texts_to_embed)}")
                response = self._client.embeddings(model=self._model, prompt=text)
                self._embedding_cache[text] = response['embedding']
        else:
            embedding_logger.info("All texts found in cache")
        
        # Return embeddings from cache
        return [self._embedding_cache[text] for text in texts]

    def similarity(self, embedding1: List[float], embedding2: List[float], mode: str = "cosine") -> float:
        """Compute similarity between two embeddings."""
        # Convert to numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        if mode == "cosine":
            # Compute cosine similarity
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        else:
            raise ValueError(f"Unsupported similarity mode: {mode}")

def load_embedding_model(**kwargs) -> DirectOllamaEmbedding:
    """Load the embedding model based on environment variables and kwargs."""
    model_name = kwargs.pop('model_name', os.getenv('OLLAMA_MODEL', 'bge-m3'))
    base_url = kwargs.pop('base_url', os.getenv('OLLAMA_API_BASE_URL'))
    return DirectOllamaEmbedding(
        model_name=model_name,
        base_url=base_url,
    )