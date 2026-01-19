"""Storage module for Supabase integration with pgvector."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import create_client, Client

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseStorage:
    """Storage service for articles and embeddings in Supabase."""
    
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self.table_name = "articles"
    
    def store_article(
        self,
        title: str,
        content: str,
        url: str,
        source: str,
        embedding: List[float],
        published_date: Optional[datetime] = None,
        author: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Store an article with its embedding."""
        try:
            data = {
                "title": title,
                "content": content,
                "url": url,
                "source": source,
                "embedding": embedding,
                "published_date": published_date.isoformat() if published_date else datetime.now().isoformat(),
                "author": author,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat()
            }
            
            result = self.client.table(self.table_name).insert(data).execute()
            logger.info(f"Stored article: {title}")
            return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Error storing article: {e}")
            return None
    
    def store_articles_batch(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Store multiple articles in batch using upsert to handle duplicates."""
        if not articles:
            return []

        try:
            # Deduplicate by URL within the batch (keep first occurrence)
            seen_urls = set()
            unique_articles = []
            for article in articles:
                if article["url"] not in seen_urls:
                    seen_urls.add(article["url"])
                    unique_articles.append(article)

            if len(unique_articles) < len(articles):
                logger.info(f"Removed {len(articles) - len(unique_articles)} duplicates within batch")

            # Use upsert to handle any remaining duplicates gracefully
            result = self.client.table(self.table_name).upsert(
                unique_articles,
                on_conflict="url"
            ).execute()
            logger.info(f"Stored {len(unique_articles)} articles")
            return result.data or []

        except Exception as e:
            logger.error(f"Error storing articles batch: {e}")
            return []
    
    def search_similar_articles(
        self,
        embedding: List[float],
        threshold: float = 0.85,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar articles using vector similarity.
        
        Note: This requires pgvector extension and a proper index on the embedding column.
        The actual RPC function needs to be created in Supabase.
        """
        try:
            # Call RPC function for vector similarity search
            # This assumes you have created an RPC function in Supabase
            result = self.client.rpc(
                "match_articles",
                {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "match_count": limit
                }
            ).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Error searching similar articles: {e}")
            return []
    
    def get_articles(
        self,
        limit: int = 100,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve articles with optional filters."""
        try:
            query = self.client.table(self.table_name).select("*")
            
            if source:
                query = query.eq("source", source)
            
            if start_date:
                query = query.gte("published_date", start_date.isoformat())
            
            query = query.order("published_date", desc=True).limit(limit)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            return []
    
    def delete_article(self, article_id: int) -> bool:
        """Delete an article by ID."""
        try:
            self.client.table(self.table_name).delete().eq("id", article_id).execute()
            logger.info(f"Deleted article with ID: {article_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting article: {e}")
            return False

    def get_existing_urls(self, urls: List[str]) -> set[str]:
        """Check which URLs already exist in the database.

        Args:
            urls: List of URLs to check

        Returns:
            Set of URLs that already exist in the database
        """
        if not urls:
            return set()

        try:
            result = self.client.table(self.table_name)\
                .select("url")\
                .in_("url", urls)\
                .execute()

            existing = {row["url"] for row in result.data} if result.data else set()
            logger.info(f"Found {len(existing)} existing URLs out of {len(urls)}")
            return existing

        except Exception as e:
            logger.error(f"Error checking existing URLs: {e}")
            return set()
    
    def create_schema_sql(self) -> str:
        """Generate SQL for creating the articles table with pgvector.
        
        This SQL should be run in Supabase SQL editor.
        """
        return """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create articles table
CREATE TABLE IF NOT EXISTS articles (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    embedding vector(1536),
    published_date TIMESTAMP WITH TIME ZONE,
    author TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS articles_embedding_idx ON articles 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create RPC function for similarity search
CREATE OR REPLACE FUNCTION match_articles(
    query_embedding vector(1536),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    id BIGINT,
    title TEXT,
    content TEXT,
    url TEXT,
    source TEXT,
    published_date TIMESTAMP WITH TIME ZONE,
    author TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        id,
        title,
        content,
        url,
        source,
        published_date,
        author,
        1 - (embedding <=> query_embedding) AS similarity
    FROM articles
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
"""


def main():
    """Print schema SQL for setup."""
    storage = SupabaseStorage()
    print("Run this SQL in your Supabase SQL editor:")
    print("=" * 80)
    print(storage.create_schema_sql())
    print("=" * 80)


if __name__ == "__main__":
    main()
