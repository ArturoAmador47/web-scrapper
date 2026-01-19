"""Configuration management for the tech news aggregator."""

from urllib.parse import urlparse
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-ada-002"
    
    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False  # Should be False in production
    api_key: str = ""  # Optional: Set to enable API authentication
    
    # Scraping
    user_agent: str = "Mozilla/5.0 (compatible; TechNewsAggregator/1.0)"
    request_timeout: int = 30
    max_retries: int = 3
    news_sources: str = ""
    
    # Similarity
    similarity_threshold: float = 0.85
    
    # Output
    output_dir: str = "./output"
    pdf_title: str = "Tech News Digest"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def get_news_sources(self) -> list[str]:
        """Get news sources as a list."""
        if not self.news_sources:
            return []
        sources = []
        for s in self.news_sources.split(","):
            s = s.strip()
            # Skip empty strings and comments
            if s and not s.startswith("#"):
                sources.append(s)
        return sources

    def get_allowed_domains(self) -> set[str]:
        """Extract allowed domains from configured news sources.

        Returns:
            Set of allowed domain names (e.g., {'techcrunch.com', 'github.blog'})
        """
        domains = set()
        for source in self.get_news_sources():
            try:
                parsed = urlparse(source)
                if parsed.netloc:
                    # Normalize domain (lowercase, remove www prefix)
                    domain = parsed.netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                    domains.add(domain)
            except Exception:
                continue
        return domains


# Global settings instance
settings = Settings()
