"""Simplified embeddings service.

This module provides core embedding generation functionality.
For clustering and content generation, see the separate modules:
- clustering.py: Article grouping by similarity
- content_generator.py: LLM content generation
- prompts.py: LLM prompt catalog
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from src.config import settings
from src.security import (
    safe_log_error,
    sanitize_text_for_llm,
    sanitize_article_data,
)

# Imports for backward compatibility
from src.embeddings.clustering import (
    cluster_articles as _cluster_articles,
    cosine_similarity,
    find_similar_articles,
)
from src.embeddings.content_generator import ContentGenerator

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Service for generating and comparing embeddings using OpenAI.
    
    This service focuses on:
    - Individual and batch embedding generation
    - Similarity comparison between embeddings
    - Duplicate detection
    
    For clustering and content generation, use the
    clustering.py and content_generator.py modules respectively.
    """
    
    def __init__(self):
        """Initialize the embeddings service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self._content_generator: Optional[ContentGenerator] = None
    
    @property
    def content_generator(self) -> ContentGenerator:
        """Lazy-loaded content generator."""
        if self._content_generator is None:
            self._content_generator = ContentGenerator()
        return self._content_generator
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to process
            
        Returns:
            Embedding vector or None on error
        """
        try:
            # Sanitizar texto
            text = sanitize_text_for_llm(text)
            
            if not text:
                logger.warning("Empty text after sanitization")
                return None
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Embedding generated: {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            safe_log_error(logger, "Error generating embedding", e)
            return None
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size to avoid rate limits
            
        Returns:
            List of embeddings (None for failed texts)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Sanitize texts
                batch = [sanitize_text_for_llm(text) for text in batch]
                # Replace empty with space
                batch = [text if text else " " for text in batch]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                safe_log_error(logger, "Error in embeddings batch", e)
                embeddings.extend([None] * len(batch))
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First vector
            embedding2: Second vector
            
        Returns:
            Similarity between -1 and 1
        """
        return cosine_similarity(embedding1, embedding2)
    
    def is_duplicate(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: Optional[float] = None
    ) -> bool:
        """Check if two embeddings represent duplicate content.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold (default from settings)
            
        Returns:
            True if duplicates
        """
        if threshold is None:
            threshold = settings.similarity_threshold
        
        similarity = self.cosine_similarity(embedding1, embedding2)
        return similarity >= threshold
    
    def find_duplicates(
        self,
        embeddings: List[List[float]],
        threshold: Optional[float] = None
    ) -> List[List[int]]:
        """Find duplicate articles based on embeddings.
        
        Args:
            embeddings: List of embeddings
            threshold: Similarity threshold
            
        Returns:
            List of duplicate groups (indices)
        """
        if threshold is None:
            threshold = settings.similarity_threshold
        
        n = len(embeddings)
        visited = set()
        duplicate_groups = []
        
        for i in range(n):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                
                similarity = self.cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups
    
    def cluster_articles(
        self,
        articles: List[Dict[str, Any]],
        max_clusters: int = 8,
        distance_threshold: float = 1.2
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group articles by semantic similarity.
        
        Args:
            articles: List of articles with 'embedding'
            max_clusters: Maximum number of clusters
            distance_threshold: Distance threshold
            
        Returns:
            Dictionary cluster_id -> list of articles
        """
        clusters = _cluster_articles(
            articles,
            n_clusters=max_clusters,
            distance_threshold=distance_threshold
        )
        
        return {cluster_id: arts for cluster_id, arts in clusters}
    
    async def cluster_and_name_articles(
        self,
        articles: List[Dict[str, Any]],
        max_clusters: int = 8
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group articles and generate descriptive cluster names.
        
        Args:
            articles: List of articles with embeddings
            max_clusters: Maximum number of clusters
            
        Returns:
            Dictionary cluster_name -> list of articles
        """
        # First group
        clusters = self.cluster_articles(articles, max_clusters=max_clusters)
        
        if not clusters:
            return {}
        
        # Generate names for each cluster
        named_clusters = {}
        for cluster_id, cluster_articles in clusters.items():
            name = self.content_generator.generate_cluster_name(cluster_articles)
            named_clusters[name] = cluster_articles
        
        logger.info(f"Grouped {len(articles)} articles into {len(named_clusters)} clusters")
        return named_clusters
    
    def generate_executive_summary(
        self,
        grouped_articles: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[str]:
        """Generate executive summary of the day.
        
        Args:
            grouped_articles: Articles grouped by topic
            
        Returns:
            Executive summary
        """
        return self.content_generator.generate_executive_summary(grouped_articles)
    
    def select_top_articles(
        self,
        articles: List[Dict[str, Any]],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """Select the most important articles.
        
        Args:
            articles: List of candidates
            count: Number to select
            
        Returns:
            Selected articles
        """
        return self.content_generator.select_top_articles(articles, count)
    
    def generate_section_brief(
        self,
        topic: str,
        articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate brief for a section.
        
        Args:
            topic: Topic name
            articles: Articles in the section
            
        Returns:
            Brief of 2-3 sentences
        """
        return self.content_generator.generate_section_brief(topic, articles)
    
    def generate_section_narrative(
        self,
        topic: str,
        articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate narrative for podcast.
        
        Args:
            topic: Topic name
            articles: Articles in the section
            
        Returns:
            Narrative of 2-4 paragraphs
        """
        return self.content_generator.generate_section_narrative(topic, articles)
    
    def enrich_grouped_articles(
        self,
        grouped_articles: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Enrich grouped articles with generated content.
        
        Args:
            grouped_articles: Articles grouped by topic
            
        Returns:
            Enriched structure with summary, tops and sections
        """
        return self.content_generator.enrich_grouped_articles(grouped_articles)


# =============================================================================
# Utility functions for compatibility
# =============================================================================

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Convenience function to generate an embedding."""
    service = EmbeddingsService()
    return await service.generate_embedding(text)


async def generate_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Convenience function to generate embeddings in batch."""
    service = EmbeddingsService()
    return await service.generate_embeddings_batch(texts)


# =============================================================================
# Main for testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        service = EmbeddingsService()
        
        # Basic test
        test_text = "Artificial intelligence is transforming the tech industry"
        embedding = await service.generate_embedding(test_text)
        
        if embedding:
            print(f"✓ Embedding generated: {len(embedding)} dimensions")
        else:
            print("✗ Error generating embedding")
    
    asyncio.run(main())
