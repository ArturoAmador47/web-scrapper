"""Scraper module for fetching tech news from various sources."""

import asyncio
import feedparser
import ipaddress
import requests
import socket
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
import logging

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Private IP ranges that should be blocked (SSRF protection)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),    # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),   # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local (AWS metadata)
    ipaddress.ip_network("0.0.0.0/8"),        # Current network
    ipaddress.ip_network("224.0.0.0/4"),      # Multicast
    ipaddress.ip_network("240.0.0.0/4"),      # Reserved
]

# Allowed URL schemes
ALLOWED_SCHEMES = {"https"}

# Cache for allowed domains (populated from settings)
_allowed_domains_cache: set[str] = set()


def get_allowed_domains() -> set[str]:
    """Get cached allowed domains from settings."""
    global _allowed_domains_cache
    if not _allowed_domains_cache:
        _allowed_domains_cache = settings.get_allowed_domains()
    return _allowed_domains_cache


def is_domain_whitelisted(hostname: str) -> bool:
    """Check if hostname is in the whitelist of allowed domains.

    Args:
        hostname: The hostname to check

    Returns:
        True if domain is whitelisted or whitelist is empty (allow all)
    """
    allowed = get_allowed_domains()

    # If no whitelist configured, allow all (for backwards compatibility)
    if not allowed:
        return True

    # Normalize hostname
    hostname = hostname.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Check exact match
    if hostname in allowed:
        return True

    # Check if it's a subdomain of an allowed domain
    for domain in allowed:
        if hostname.endswith("." + domain):
            return True

    return False


def is_safe_url(url: str, check_whitelist: bool = True) -> tuple[bool, str]:
    """Validate URL for SSRF protection and domain whitelist.

    Args:
        url: URL to validate
        check_whitelist: Whether to check domain against whitelist (default True)

    Returns:
        Tuple of (is_safe, reason)
    """
    if not url:
        return False, "Empty URL"

    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return False, f"Blocked scheme: {parsed.scheme}"

        # Check for empty host
        if not parsed.netloc:
            return False, "Missing host"

        # Extract hostname
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid hostname"

        # Block localhost variations
        if hostname.lower() in ("localhost", "localhost.localdomain"):
            return False, "Localhost blocked"

        # Check domain whitelist (if enabled and whitelist is configured)
        if check_whitelist and not is_domain_whitelisted(hostname):
            return False, f"Domain not whitelisted: {hostname}"

        # Try to resolve hostname to IP
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)

            # Check against blocked ranges
            for blocked_range in BLOCKED_IP_RANGES:
                if ip in blocked_range:
                    return False, f"Blocked IP range: {ip_str}"

        except socket.gaierror:
            # Could not resolve - might be valid external domain
            # Allow it but log for monitoring
            logger.warning(f"Could not resolve hostname: {hostname}")

        return True, "OK"

    except Exception as e:
        return False, f"URL validation error: {str(e)}"


def validate_url(url: str) -> str:
    """Validate and return URL if safe, raise ValueError otherwise.

    Args:
        url: URL to validate

    Returns:
        The validated URL

    Raises:
        ValueError: If URL is not safe
    """
    is_safe, reason = is_safe_url(url)
    if not is_safe:
        logger.warning(f"Blocked unsafe URL: {url} - Reason: {reason}")
        raise ValueError(f"Unsafe URL blocked: {reason}")
    return url


class Article:
    """Represents a news article."""
    
    def __init__(
        self,
        title: str,
        content: str,
        url: str,
        source: str,
        published_date: Optional[datetime] = None,
        author: Optional[str] = None
    ):
        self.title = title
        self.content = content
        self.url = url
        self.source = source
        self.published_date = published_date or datetime.now()
        self.author = author
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "author": self.author
        }


class NewsScraper:
    """Scraper for tech news from RSS feeds and web pages."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": settings.user_agent
        }
        self.timeout = settings.request_timeout
    
    async def scrape_rss_feed(self, feed_url: str) -> List[Article]:
        """Scrape articles from an RSS feed.

        Args:
            feed_url: URL of the RSS feed (must be http/https, no private IPs)

        Returns:
            List of Article objects
        """
        logger.info(f"Scraping RSS feed: {feed_url}")
        articles = []

        # SSRF Protection: Validate URL before fetching
        try:
            validate_url(feed_url)
        except ValueError as e:
            logger.error(f"SSRF Protection - Blocked feed URL: {feed_url} - {e}")
            return articles

        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:10]:  # Limit to 10 recent articles
                title = entry.get("title", "No Title")
                link = entry.get("link", "")
                
                # Get content
                content = ""
                if hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description
                elif hasattr(entry, "content"):
                    content = entry.content[0].value if entry.content else ""
                
                # Get published date
                published_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6])
                
                # Get author
                author = entry.get("author", None)
                
                # Determine source
                source = feed.feed.get("title", feed_url)
                
                article = Article(
                    title=title,
                    content=content,
                    url=link,
                    source=source,
                    published_date=published_date,
                    author=author
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error scraping RSS feed {feed_url}: {e}")
        
        logger.info(f"Scraped {len(articles)} articles from {feed_url}")
        return articles
    
    async def scrape_webpage(self, url: str) -> Optional[str]:
        """Scrape content from a webpage using Crawl4AI or fallback to requests.

        Args:
            url: URL of the webpage (must be http/https, no private IPs)

        Returns:
            Page content as string, or None if failed
        """
        logger.info(f"Scraping webpage: {url}")

        # SSRF Protection: Validate URL before fetching
        try:
            validate_url(url)
        except ValueError as e:
            logger.error(f"SSRF Protection - Blocked webpage URL: {url} - {e}")
            return None

        try:
            if CRAWL4AI_AVAILABLE:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    if result.success:
                        return result.markdown
            
            # Fallback to requests
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.text
            
        except Exception as e:
            logger.error(f"Error scraping webpage {url}: {e}")
            return None
    
    async def scrape_all_sources(self, sources: Optional[List[str]] = None) -> List[Article]:
        """Scrape all configured news sources."""
        if sources is None:
            sources = settings.get_news_sources()
        
        if not sources:
            logger.warning("No news sources configured")
            return []
        
        logger.info(f"Scraping {len(sources)} news sources")
        
        all_articles = []
        tasks = [self.scrape_rss_feed(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in scraping task: {result}")
        
        logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles


async def main():
    """Test the scraper."""
    scraper = NewsScraper()
    articles = await scraper.scrape_all_sources()
    for article in articles[:5]:
        print(f"\nTitle: {article.title}")
        print(f"Source: {article.source}")
        print(f"URL: {article.url}")
        print(f"Date: {article.published_date}")
        print(f"Content preview: {article.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
