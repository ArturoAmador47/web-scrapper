"""FastAPI application for the news aggregator."""

import os
import re
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from pathlib import Path

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Regex for safe PDF filenames: alphanumeric, hyphens, underscores, dots
SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+\.pdf$')

from src.aggregator import NewsAggregator
from src.config import settings
from src.storage.supabase_storage import SupabaseStorage
from src.security import safe_log_error, get_safe_error_detail, verify_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Tech News Aggregator API",
    description="Automated tech news aggregation with AI-powered deduplication",
    version="1.0.0"
)

# CORS configuration - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.api_key else [],  # Restrict if auth enabled
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


class WebhookRequest(BaseModel):
    """Request model for n8n webhook with input validation."""
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
        description="Whether to group articles by topic in the PDF"
    )
    enrich: bool = Field(
        True,
        description="Add executive summary, top 3 articles, and section briefs"
    )

    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate that sources are valid HTTP/HTTPS URLs."""
        if v is None:
            return v

        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'[a-zA-Z0-9]'  # domain must start with alphanumeric
            r'[a-zA-Z0-9\-\.]*'  # rest of domain
            r'[a-zA-Z0-9]'  # domain must end with alphanumeric
            r'(?::\d+)?'  # optional port
            r'(?:/[^\s]*)?$'  # optional path
        )

        validated = []
        for url in v:
            if not isinstance(url, str):
                raise ValueError(f"Source must be a string, got {type(url)}")
            url = url.strip()
            if not url:
                continue
            if not url_pattern.match(url):
                raise ValueError(f"Invalid URL format: {url}")
            if len(url) > 2048:
                raise ValueError(f"URL too long: {url[:50]}...")
            validated.append(url)

        return validated if validated else None


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
@limiter.limit("5/minute")
async def scrape_news(
    request: Request,
    scrape_request: ScrapeRequest,
    _api_key: str = Depends(verify_api_key)
):
    """
    Scrape news articles from configured sources.

    This endpoint triggers the full pipeline:
    1. Scrapes articles from RSS feeds
    2. Filters out existing articles (if store=True)
    3. Generates embeddings
    4. Deduplicates articles (optional)
    5. Stores in Supabase (optional)
    6. Clusters by topic and generates PDF digest (optional)
    
    Requires X-API-Key header if API_KEY is configured.
    """
    try:
        logger.info(f"Received scrape request: deduplicate={scrape_request.deduplicate}, store={scrape_request.store}")

        result = await aggregator.run_full_pipeline(
            sources=scrape_request.sources,
            deduplicate=scrape_request.deduplicate,
            store=scrape_request.store,
            generate_pdf=scrape_request.generate_pdf,
            group_by_topic=scrape_request.group_by_topic,
            enrich=scrape_request.enrich
        )

        return PipelineResponse(**result)

    except Exception as e:
        safe_log_error(logger, "Error in scrape endpoint", e)
        raise HTTPException(status_code=500, detail=get_safe_error_detail(e))


@app.get("/articles", response_model=List[ArticleResponse])
@limiter.limit("30/minute")
async def get_articles(
    request: Request,
    limit: int = 100,
    source: Optional[str] = None,
    _api_key: str = Depends(verify_api_key)
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
        safe_log_error(logger, "Error retrieving articles", e)
        raise HTTPException(status_code=500, detail=get_safe_error_detail(e))


@app.get("/pdf/{filename}")
@limiter.limit("30/minute")
async def download_pdf(request: Request, filename: str):
    """
    Download a generated PDF file.

    Args:
        filename: Name of the PDF file (alphanumeric, hyphens, underscores only)
    """
    try:
        # Step 1: Strip any directory components (defense layer 1)
        filename = os.path.basename(filename)

        # Step 2: Validate filename against safe pattern (defense layer 2)
        if not SAFE_FILENAME_PATTERN.match(filename):
            logger.warning(f"Rejected unsafe filename: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename format")

        # Step 3: Construct and resolve paths
        output_dir = Path(settings.output_dir).resolve()
        pdf_path = (output_dir / filename).resolve()

        # Step 4: Verify the resolved path is within output directory (defense layer 3)
        try:
            pdf_path.relative_to(output_dir)
        except ValueError:
            logger.warning(f"Path traversal attempt blocked: {filename}")
            raise HTTPException(status_code=400, detail="Invalid file path")

        # Step 5: Check file exists
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
        safe_log_error(logger, "Error downloading PDF", e)
        raise HTTPException(status_code=500, detail=get_safe_error_detail(e))


@app.get("/pdfs")
@limiter.limit("30/minute")
async def list_pdfs(request: Request):
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
        safe_log_error(logger, "Error listing PDFs", e)
        raise HTTPException(status_code=500, detail=get_safe_error_detail(e))


@app.post("/webhook/n8n", response_model=PipelineResponse)
@limiter.limit("5/minute")
async def n8n_webhook(
    request: Request,
    webhook_request: WebhookRequest,
    _api_key: str = Depends(verify_api_key)
):
    """
    Webhook endpoint for n8n workflows.

    Accepts a validated JSON payload with scraping configuration and triggers the pipeline.
    All inputs are validated using Pydantic models.
    Requires X-API-Key header if API_KEY is configured.
    """
    try:
        logger.info(f"Received n8n webhook: sources={len(webhook_request.sources) if webhook_request.sources else 'default'}")

        # Run pipeline with validated parameters
        result = await aggregator.run_full_pipeline(
            sources=webhook_request.sources,
            deduplicate=webhook_request.deduplicate,
            store=webhook_request.store,
            generate_pdf=webhook_request.generate_pdf,
            group_by_topic=webhook_request.group_by_topic,
            enrich=webhook_request.enrich
        )

        return PipelineResponse(**result)

    except Exception as e:
        safe_log_error(logger, "Error in n8n webhook", e)
        raise HTTPException(status_code=500, detail=get_safe_error_detail(e))


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
