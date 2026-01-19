"""Main orchestration module for the news aggregator."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.scraper.news_scraper import NewsScraper, Article
from src.embeddings.embeddings_service import EmbeddingsService
from src.storage.supabase_storage import SupabaseStorage
from src.pdf_generator.pdf_service import PDFGenerator
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsAggregator:
    """Main orchestrator for the news aggregation pipeline."""
    
    def __init__(self):
        self.scraper = NewsScraper()
        self.embeddings_service = EmbeddingsService()
        self.storage = SupabaseStorage()
        self.pdf_generator = PDFGenerator()
    
    async def process_articles(
        self,
        articles: List[Article],
        deduplicate: bool = True
    ) -> List[Dict[str, Any]]:
        """Process articles: generate embeddings and deduplicate."""
        if not articles:
            logger.warning("No articles to process")
            return []
        
        logger.info(f"Processing {len(articles)} articles")
        
        # Generate embeddings
        texts = [f"{article.title} {article.content}" for article in articles]
        embeddings = await self.embeddings_service.generate_embeddings_batch(texts)
        
        # Create article dictionaries with embeddings
        processed_articles = []
        for article, embedding in zip(articles, embeddings):
            if embedding:
                processed_articles.append({
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "source": article.source,
                    "published_date": article.published_date.isoformat(),
                    "author": article.author,
                    "embedding": embedding
                })
        
        logger.info(f"Generated embeddings for {len(processed_articles)} articles")
        
        # Deduplicate if requested
        if deduplicate and len(processed_articles) > 1:
            processed_articles = self._deduplicate_articles(processed_articles)
        
        return processed_articles
    
    def _deduplicate_articles(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on embedding similarity."""
        if len(articles) <= 1:
            return articles
        
        embeddings = [article["embedding"] for article in articles]
        duplicate_groups = self.embeddings_service.find_duplicates(embeddings)
        
        # Keep only the first article from each duplicate group
        duplicates_to_remove = set()
        for group in duplicate_groups:
            # Remove all but the first article in the group
            duplicates_to_remove.update(group[1:])
        
        unique_articles = [
            article for idx, article in enumerate(articles)
            if idx not in duplicates_to_remove
        ]
        
        logger.info(
            f"Deduplication: {len(articles)} -> {len(unique_articles)} "
            f"({len(articles) - len(unique_articles)} duplicates removed)"
        )
        
        return unique_articles
    
    def store_articles(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Store articles in Supabase."""
        if not articles:
            return []
        
        stored = self.storage.store_articles_batch(articles)
        logger.info(f"Stored {len(stored)} articles in database")
        return stored
    
    async def generate_digest(
        self,
        articles: List[Dict[str, Any]],
        filename: Optional[str] = None,
        group_by_topic: bool = True,
        enrich: bool = True
    ) -> str:
        """Generate PDF digest from articles.

        Args:
            articles: List of processed articles with embeddings
            filename: Optional output filename
            group_by_topic: If True, cluster articles by topic and generate index
            enrich: If True, add executive summary, top picks, and section briefs
        """
        if not articles:
            logger.warning("No articles to generate digest from")
            return None

        if group_by_topic and len(articles) >= 3:
            # Cluster articles by topic
            logger.info("Clustering articles by topic...")
            grouped_articles = await self.embeddings_service.cluster_and_name_articles(
                articles,
                max_clusters=8
            )

            if enrich:
                # Enrich with executive summary, top picks, and briefs
                logger.info("Enriching content with AI summaries...")
                enriched_data = self.embeddings_service.enrich_grouped_articles(
                    grouped_articles
                )
                pdf_path = self.pdf_generator.generate_pdf_enriched(enriched_data, filename)
                logger.info(f"Generated enriched digest PDF: {pdf_path}")
            else:
                pdf_path = self.pdf_generator.generate_pdf_grouped(grouped_articles, filename)
                logger.info(f"Generated grouped digest PDF: {pdf_path}")
        else:
            # Generate flat PDF
            pdf_path = self.pdf_generator.generate_pdf(articles, filename)
            logger.info(f"Generated digest PDF: {pdf_path}")

        return str(pdf_path)
    
    def _filter_existing_articles(self, articles: List[Article]) -> List[Article]:
        """Filter out articles that already exist in the database.

        Args:
            articles: List of scraped articles

        Returns:
            List of articles that don't exist in the database
        """
        if not articles:
            return []

        urls = [article.url for article in articles]
        existing_urls = self.storage.get_existing_urls(urls)

        if not existing_urls:
            return articles

        new_articles = [a for a in articles if a.url not in existing_urls]
        logger.info(
            f"Filtered existing articles: {len(articles)} -> {len(new_articles)} "
            f"({len(existing_urls)} already in database)"
        )
        return new_articles

    async def run_full_pipeline(
        self,
        sources: Optional[List[str]] = None,
        deduplicate: bool = True,
        store: bool = True,
        generate_pdf: bool = True,
        group_by_topic: bool = True,
        enrich: bool = True
    ) -> Dict[str, Any]:
        """Run the complete news aggregation pipeline.

        Args:
            sources: Optional list of RSS feed URLs to scrape
            deduplicate: Remove duplicate articles based on similarity
            store: Store articles in Supabase database
            generate_pdf: Generate PDF digest
            group_by_topic: Cluster articles by topic in PDF (requires generate_pdf=True)
            enrich: Add executive summary, top 3 picks, and section briefs (requires group_by_topic=True)

        Returns:
            Dictionary with pipeline results including article count and PDF path.
        """
        logger.info("Starting news aggregation pipeline")
        start_time = datetime.now()

        # Step 1: Scrape articles
        articles = await self.scraper.scrape_all_sources(sources)
        total_scraped = len(articles)

        if not articles:
            logger.warning("No articles scraped")
            return {
                "success": False,
                "message": "No articles found",
                "articles_scraped": 0
            }

        # Step 2: Filter out articles that already exist in database
        if store:
            articles = self._filter_existing_articles(articles)
            if not articles:
                logger.info("All articles already exist in database")
                return {
                    "success": True,
                    "message": "All articles already exist in database",
                    "articles_scraped": total_scraped,
                    "articles_new": 0,
                    "articles_processed": 0,
                    "articles_stored": 0
                }

        # Step 3: Process articles (embeddings + deduplication)
        processed_articles = await self.process_articles(articles, deduplicate)

        # Step 4: Store articles
        stored_articles = []
        if store and processed_articles:
            stored_articles = self.store_articles(processed_articles)

        # Step 5: Generate PDF (with optional topic clustering and enrichment)
        pdf_path = None
        if generate_pdf and processed_articles:
            pdf_path = await self.generate_digest(
                processed_articles,
                group_by_topic=group_by_topic,
                enrich=enrich
            )

        elapsed_time = (datetime.now() - start_time).total_seconds()

        result = {
            "success": True,
            "articles_scraped": total_scraped,
            "articles_new": len(articles),
            "articles_processed": len(processed_articles),
            "articles_stored": len(stored_articles),
            "pdf_path": pdf_path,
            "elapsed_time": elapsed_time
        }
        
        logger.info(f"Pipeline completed in {elapsed_time:.2f}s: {result}")
        return result


async def main():
    """Test the aggregator."""
    aggregator = NewsAggregator()
    result = await aggregator.run_full_pipeline(
        deduplicate=True,
        store=False,  # Set to True if Supabase is configured
        generate_pdf=True
    )
    print("\nPipeline Result:")
    print(f"Success: {result['success']}")
    print(f"Articles scraped: {result.get('articles_scraped', 0)}")
    print(f"Articles processed: {result.get('articles_processed', 0)}")
    print(f"PDF path: {result.get('pdf_path', 'N/A')}")
    print(f"Time: {result.get('elapsed_time', 0):.2f}s")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
