"""FastAPI application for the news aggregator."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from pathlib import Path

from src.aggregator import NewsAggregator
from src.config import settings
from src.storage.supabase_storage import SupabaseStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tech News Aggregator API",
    description="Automated tech news aggregation with AI-powered deduplication",
    version="1.0.0"
)

# Global instances
aggregator = NewsAggregator()
storage = SupabaseStorage()


class ScrapeRequest(BaseModel):
    """Request model for scraping."""
    sources: Optional[List[str]] = Field(
        None,
        description="List of RSS feed URLs. If not provided, uses configured sources."
    )
    deduplicate: bool = Field(
        True,
        description="Whether to deduplicate articles using embeddings"
    )
    store: bool = Field(
        True,
        description="Whether to store articles in Supabase"
    )
    generate_pdf: bool = Field(
        True,
        description="Whether to generate PDF digest"
    )
    group_by_topic: bool = Field(
        True,
        description="Whether to group articles by topic in the PDF (uses AI clustering)"
    )
    enrich: bool = Field(
        True,
        description="Add executive summary, top 3 must-read articles, and section briefs (optimized for NotebookLM)"
    )


class ArticleResponse(BaseModel):
    """Response model for article."""
    title: str
    content: str
    url: str
    source: str
    published_date: str
    author: Optional[str] = None


class PipelineResponse(BaseModel):
    """Response model for pipeline execution."""
    success: bool
    message: Optional[str] = None
    articles_scraped: int
    articles_new: Optional[int] = None
    articles_processed: int
    articles_stored: int
    pdf_path: Optional[str] = None
    elapsed_time: float


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Tech News Aggregator API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/scrape", response_model=PipelineResponse)
async def scrape_news(request: ScrapeRequest):
    """
    Scrape news articles from configured sources.

    This endpoint triggers the full pipeline:
    1. Scrapes articles from RSS feeds
    2. Filters out existing articles (if store=True)
    3. Generates embeddings
    4. Deduplicates articles (optional)
    5. Stores in Supabase (optional)
    6. Clusters by topic and generates PDF digest (optional)
    """
    try:
        logger.info(f"Received scrape request: {request}")

        result = await aggregator.run_full_pipeline(
            sources=request.sources,
            deduplicate=request.deduplicate,
            store=request.store,
            generate_pdf=request.generate_pdf,
            group_by_topic=request.group_by_topic,
            enrich=request.enrich
        )

        return PipelineResponse(**result)

    except Exception as e:
        logger.error(f"Error in scrape endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/articles", response_model=List[ArticleResponse])
async def get_articles(
    limit: int = 100,
    source: Optional[str] = None
):
    """
    Retrieve stored articles from Supabase.
    
    Args:
        limit: Maximum number of articles to return
        source: Filter by source name
    """
    try:
        articles = storage.get_articles(limit=limit, source=source)
        return articles
        
    except Exception as e:
        logger.error(f"Error retrieving articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdf/{filename}")
async def download_pdf(filename: str):
    """
    Download a generated PDF file.
    
    Args:
        filename: Name of the PDF file
    """
    try:
        # Sanitize filename to prevent path traversal
        import os
        filename = os.path.basename(filename)
        
        # Ensure filename ends with .pdf
        if not filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        pdf_path = Path(settings.output_dir) / filename
        
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")
        
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdfs")
async def list_pdfs():
    """
    List all generated PDF files.
    """
    try:
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(output_dir.glob("*.pdf"))
        
        files = []
        for pdf_file in pdf_files:
            stat = pdf_file.stat()
            files.append({
                "filename": pdf_file.name,
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "download_url": f"/pdf/{pdf_file.name}"
            })
        
        return {
            "count": len(files),
            "files": files
        }
        
    except Exception as e:
        logger.error(f"Error listing PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/n8n")
async def n8n_webhook(data: Dict[str, Any]):
    """
    Webhook endpoint for n8n workflows.

    Accepts a JSON payload with scraping configuration and triggers the pipeline.
    """
    try:
        logger.info(f"Received n8n webhook: {data}")

        # Extract parameters from webhook data
        sources = data.get("sources")
        deduplicate = data.get("deduplicate", True)
        store = data.get("store", True)
        generate_pdf = data.get("generate_pdf", True)
        group_by_topic = data.get("group_by_topic", True)
        enrich = data.get("enrich", True)

        # Run pipeline
        result = await aggregator.run_full_pipeline(
            sources=sources,
            deduplicate=deduplicate,
            store=store,
            generate_pdf=generate_pdf,
            group_by_topic=group_by_topic,
            enrich=enrich
        )

        return result

    except Exception as e:
        logger.error(f"Error in n8n webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """
    Get current configuration (excluding sensitive data).
    """
    return {
        "embedding_model": settings.embedding_model,
        "similarity_threshold": settings.similarity_threshold,
        "news_sources": settings.get_news_sources(),
        "output_dir": settings.output_dir,
        "pdf_title": settings.pdf_title
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
