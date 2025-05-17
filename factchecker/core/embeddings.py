"""
Module for loading and configuring embeddings using either Ollama or local HuggingFace models.
"""

import os
from typing import Optional, List, Dict, Any, Literal
import logging
import logging.handlers
from ollama import Client as OllamaClient
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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

class BaseEmbedding:
    """Base class for embedding implementations."""
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        raise NotImplementedError
        
    def get_text_embeddings(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        raise NotImplementedError
        
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

class DirectOllamaEmbedding(BaseEmbedding):
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

class HuggingFaceBGEEmbedding(BaseEmbedding):
    """Local HuggingFace BGE embedding class."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        """Initialize with local HuggingFace model."""
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        self._embedding_cache: Dict[str, List[float]] = {}
        embedding_logger.info(f"HuggingFaceBGEEmbedding initialized with model {model_name} on {self._device}")
        
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embedding_logger.info("ENTRY: get_text_embedding called for single text")
        # Check cache first
        if text in self._embedding_cache:
            embedding_logger.info("Cache hit for single text embedding")
            return self._embedding_cache[text]
            
        embedding_logger.info(f"Getting single embedding for text of length {len(text)}")
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()  # Use CLS token
            
        embedding = embeddings[0].tolist()
        self._embedding_cache[text] = embedding
        embedding_logger.info("Successfully got single embedding")
        return embedding

    def get_text_embeddings(self, texts: List[str], show_progress: bool = True, batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts with batching."""
        embedding_logger.info(f"ENTRY: get_text_embeddings called with {len(texts)} texts")
        # Check which texts need to be embedded
        texts_to_embed = [text for text in texts if text not in self._embedding_cache]
        
        if texts_to_embed:
            embedding_logger.info(f"Getting embeddings for {len(texts_to_embed)} texts")
            # Process in batches
            for i in range(0, len(texts_to_embed), batch_size):
                if show_progress:
                    embedding_logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts_to_embed)-1)//batch_size + 1}")
                batch_texts = texts_to_embed[i:i + batch_size]
                inputs = self._tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()  # Use CLS token
                    
                for text, embedding in zip(batch_texts, embeddings):
                    self._embedding_cache[text] = embedding.tolist()
        else:
            embedding_logger.info("All texts found in cache")
        
        # Return embeddings from cache
        return [self._embedding_cache[text] for text in texts]

def load_embedding_model(
    provider: Literal["ollama", "huggingface"] = "ollama",
    **kwargs
) -> BaseEmbedding:
    """Load the embedding model based on provider and kwargs."""
    if provider == "ollama":
        # Always use bge-m3 for embeddings, ignore environment variable
        model_name = kwargs.pop('model_name', 'bge-m3')
        base_url = kwargs.pop('base_url', os.getenv('OLLAMA_API_BASE_URL'))
        return DirectOllamaEmbedding(
            model_name=model_name,
            base_url=base_url,
        )
    elif provider == "huggingface":
        model_name = kwargs.pop('model_name', 'BAAI/bge-m3')
        device = kwargs.pop('device', None)
        return HuggingFaceBGEEmbedding(
            model_name=model_name,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")