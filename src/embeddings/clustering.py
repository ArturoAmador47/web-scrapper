import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.security import safe_log_error

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity value between -1 and 1
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def cluster_articles(
    articles: List[Dict[str, Any]],
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.5,
    min_cluster_size: int = 2
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Group articles by embedding similarity.
    
    Uses AgglomerativeClustering to create hierarchical groups
    of similar articles.
    
    Args:
        articles: List of articles with 'embedding' field
        n_clusters: Fixed number of clusters (None for automatic)
        distance_threshold: Distance threshold for automatic clustering
        min_cluster_size: Minimum cluster size to be included
        
    Returns:
        List of tuples (cluster_id, articles_in_cluster)
    """
    if not articles:
        return []
    
    # Filter articles without embedding
    articles_with_embeddings = [
        a for a in articles 
        if a.get('embedding') is not None
    ]
    
    if len(articles_with_embeddings) < 2:
        # Not enough articles for clustering
        if articles_with_embeddings:
            return [(0, articles_with_embeddings)]
        return []
    
    try:
        # Create embeddings matrix
        embeddings = np.array([a['embedding'] for a in articles_with_embeddings])
        
        # Configure clustering
        if n_clusters is not None:
            # Fixed number of clusters
            n_clusters = min(n_clusters, len(articles_with_embeddings))
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
        else:
            # Automatic distance-based clustering
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average'
            )
        
        # Run clustering
        labels = clusterer.fit_predict(embeddings)
        
        # Group articles by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for article, label in zip(articles_with_embeddings, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(article)
        
        # Filter small clusters and sort by size
        result = [
            (cluster_id, articles_list)
            for cluster_id, articles_list in clusters.items()
            if len(articles_list) >= min_cluster_size
        ]
        
        # Sort by cluster size (largest first)
        result.sort(key=lambda x: len(x[1]), reverse=True)
        
        logger.info(
            f"Clustered {len(articles_with_embeddings)} articles into "
            f"{len(result)} clusters"
        )
        
        return result
        
    except Exception as e:
        safe_log_error(logger, "Clustering failed", e)
        # Fallback: todos en un solo cluster
        return [(0, articles_with_embeddings)]


def find_similar_articles(
    target_embedding: List[float],
    articles: List[Dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.7
) -> List[Tuple[Dict[str, Any], float]]:
    """Find articles similar to a given embedding.
    
    Args:
        target_embedding: Reference embedding
        articles: List of articles with embeddings
        top_k: Maximum number of results
        min_similarity: Minimum similarity to include
        
    Returns:
        List of tuples (article, similarity_score)
    """
    if not articles or not target_embedding:
        return []
    
    similarities = []
    
    for article in articles:
        embedding = article.get('embedding')
        if embedding is None:
            continue
        
        sim = cosine_similarity(target_embedding, embedding)
        if sim >= min_similarity:
            similarities.append((article, sim))
    
    # Ordenar por similitud descendente
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def merge_small_clusters(
    clusters: List[Tuple[int, List[Dict[str, Any]]]],
    min_size: int = 3
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Merge small clusters into 'Other' or with the most similar cluster.
    
    Args:
        clusters: List of clusters (id, articles)
        min_size: Minimum size to keep cluster separate
        
    Returns:
        List of clusters with small ones merged
    """
    large_clusters = []
    small_articles = []
    
    for cluster_id, articles in clusters:
        if len(articles) >= min_size:
            large_clusters.append((cluster_id, articles))
        else:
            small_articles.extend(articles)
    
    if small_articles:
        # Add "Other" cluster with small articles
        other_id = max(c[0] for c in large_clusters) + 1 if large_clusters else 0
        large_clusters.append((other_id, small_articles))
    
    return large_clusters


def get_cluster_centroid(articles: List[Dict[str, Any]]) -> Optional[List[float]]:
    """Calculate the centroid of a cluster of articles.
    
    Args:
        articles: Articles in the cluster
        
    Returns:
        Average centroid vector or None if no embeddings
    """
    embeddings = [
        a['embedding'] for a in articles 
        if a.get('embedding') is not None
    ]
    
    if not embeddings:
        return None
    
    centroid = np.mean(embeddings, axis=0)
    return centroid.tolist()
