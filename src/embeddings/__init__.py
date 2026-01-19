"""Embeddings module for article generation and comparison.

Module structure:
- embeddings_service.py: Main embeddings service
- clustering.py: Similarity-based grouping logic
- content_generator.py: LLM content generation
- prompts.py: Centralized prompt catalog

Basic usage:
    from src.embeddings import EmbeddingsService
    
    service = EmbeddingsService()
    embedding = await service.generate_embedding("text")
"""

from src.embeddings.embeddings_service import (
    EmbeddingsService,
    generate_embedding,
    generate_embeddings_batch,
)
from src.embeddings.clustering import (
    cluster_articles,
    cosine_similarity,
    find_similar_articles,
)
from src.embeddings.content_generator import ContentGenerator

__all__ = [
    # Main service
    "EmbeddingsService",
    "generate_embedding",
    "generate_embeddings_batch",
    # Clustering
    "cluster_articles",
    "cosine_similarity",
    "find_similar_articles",
    # Content generation
    "ContentGenerator",
]
