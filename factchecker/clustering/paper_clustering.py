"""Paper clustering mechanism using the configured embedding model from LlamaIndex settings."""

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClusterResult:
    """Results of paper clustering."""
    domain_papers: Dict[str, List[str]]  # Papers grouped by domain
    domain_scores: Dict[str, List[float]]  # Relevance scores for each paper in each domain

class PaperClusterer:
    """Clusters papers using the configured embedding model."""
    
    def __init__(self):
        """Initialize the paper clusterer using the embedding model from LlamaIndex settings."""
        if not Settings.embed_model:
            raise ValueError("No embedding model configured in LlamaIndex settings")
    
    def _compute_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Compute embeddings for texts using the configured embedding model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Computing embeddings for {len(texts)} texts in a single batch")
        try:
            # Use batch processing for efficiency
            embeddings = Settings.embed_model.get_text_embedding_batch(texts)
            logger.info(f"Successfully computed embeddings with shape {np.array(embeddings).shape}")
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            raise
    
    def cluster_papers_kmeans(
        self,
        papers: List[str],
        n_clusters: int,
        random_state: int = 42
    ) -> Dict[int, List[str]]:
        """Cluster papers using KMeans clustering on their embeddings.
        
        Args:
            papers: List of papers to cluster
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping cluster IDs to lists of papers
        """
        if len(papers) < n_clusters:
            logger.warning(f"Number of papers ({len(papers)}) is less than requested clusters ({n_clusters})")
            n_clusters = len(papers)
            
        logger.info(f"Clustering {len(papers)} papers into {n_clusters} groups")
        
        # Compute embeddings
        embeddings = self._compute_embeddings(papers)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group papers by cluster
        clustered_papers = {i: [] for i in range(n_clusters)}
        for paper_idx, cluster_id in enumerate(clusters):
            clustered_papers[cluster_id].append(papers[paper_idx])
        
        # Log cluster sizes
        for cluster_id, cluster_papers in clustered_papers.items():
            logger.info(f"Cluster {cluster_id}: {len(cluster_papers)} papers")
            
        return clustered_papers 