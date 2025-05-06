"""Reranker utilities for the advocate mediator system."""

from typing import List, Tuple
import requests

class BGEReranker:
    """Wrapper for BGE reranker using Ollama's BGE-Large model."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the BGE reranker.
        
        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url
    
    def rerank_papers(
        self,
        query: str,
        papers: List[str],
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """Rerank papers based on relevance to query using BGE-Large.
        
        Args:
            query: The query to compare papers against
            papers: List of paper texts to rerank
            top_k: Number of top papers to return. If None, returns all papers.
            
        Returns:
            List of (paper, score) tuples sorted by descending score
        """
        # Get scores for all query-paper pairs
        paper_scores = []
        for paper in papers:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "bge-large-en:latest",
                    "prompt": f"{query} [SEP] {paper}",
                    "mode": "hybrid",
                    "raw": True
                }
            )
            response.raise_for_status()
            scores = response.json()
            # Use combined score (70% dense, 30% sparse)
            score = 0.7 * scores["dense"] + 0.3 * scores["sparse"]
            paper_scores.append((paper, score))
        
        # Sort by score
        paper_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            paper_scores = paper_scores[:top_k]
            
        return paper_scores 