"""Security utilities for the tech news aggregator.

Provides:
- Log sanitization (remove API keys, secrets)
- LLM injection prevention (sanitize user input before sending to LLM)
- API authentication
"""

import logging
import re
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Sensitive Data Patterns (for log sanitization)
# =============================================================================

# Patterns that might contain API keys or secrets
SENSITIVE_PATTERNS = [
    # API Keys
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})', re.IGNORECASE), '[OPENAI_KEY_REDACTED]'),
    (re.compile(r'(api[_-]?key["\s:=]+)["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(bearer\s+)([a-zA-Z0-9_\-\.]+)', re.IGNORECASE), r'\1[REDACTED]'),
    
    # Supabase Keys (JWT format)
    (re.compile(r'(eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+)'), '[JWT_REDACTED]'),
    
    # Generic secrets
    (re.compile(r'(password["\s:=]+)["\']?([^\s"\']{8,})["\']?', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(secret["\s:=]+)["\']?([^\s"\']{8,})["\']?', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(token["\s:=]+)["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE), r'\1[REDACTED]'),
    
    # Connection strings
    (re.compile(r'(postgresql://[^:]+:)([^@]+)(@)', re.IGNORECASE), r'\1[REDACTED]\3'),
    (re.compile(r'(https://[^:]+:)([^@]+)(@)', re.IGNORECASE), r'\1[REDACTED]\3'),
]


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error message to remove potential secrets.
    
    Args:
        error: The exception to sanitize
        
    Returns:
        Sanitized error message safe for logging
    """
    message = str(error)
    
    for pattern, replacement in SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    
    # Truncate very long messages (might contain full responses)
    if len(message) > 500:
        message = message[:500] + "... [truncated]"
    
    return message


def safe_log_error(logger_instance: logging.Logger, context: str, error: Exception) -> None:
    """Log an error safely without exposing sensitive data.
    
    Args:
        logger_instance: The logger to use
        context: Description of what operation failed
        error: The exception that occurred
    """
    safe_message = sanitize_error_message(error)
    error_type = type(error).__name__
    logger_instance.error(f"{context}: [{error_type}] {safe_message}")


def get_safe_error_detail(error: Exception) -> str:
    """Get a safe error detail for API responses.
    
    Args:
        error: The exception
        
    Returns:
        Generic error message without sensitive details
    """
    error_type = type(error).__name__
    
    # Map common errors to user-friendly messages
    error_messages = {
        "AuthenticationError": "Authentication failed. Please check your API credentials.",
        "RateLimitError": "Rate limit exceeded. Please try again later.",
        "APIConnectionError": "Failed to connect to external service.",
        "TimeoutError": "Request timed out. Please try again.",
        "ValidationError": "Invalid request data.",
        "ValueError": "Invalid input provided.",
        "ConnectionError": "Network connection failed.",
    }
    
    return error_messages.get(error_type, f"An error occurred: {error_type}")


# =============================================================================
# API Authentication
# =============================================================================

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN


# =============================================================================
# LLM Injection Prevention
# =============================================================================

# Patterns that indicate prompt injection attempts
LLM_INJECTION_PATTERNS = [
    # System override attempts
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"disregard\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"forget\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"override\s+(system|assistant)",
    
    # Role-play/jailbreak attempts
    r"you\s+are\s+(now|going\s+to\s+be)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if|though)",
    r"from\s+now\s+on",
    
    # Common prompt delimiters
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"###\s*(System|User|Assistant)",
    
    # System revelation attempts
    r"(what|reveal|show|tell).*(system\s+prompt|instructions)",
    r"print\s+(your|the)\s+(prompt|instructions)",
    
    # Shell/code commands
    r"```\s*(python|bash|shell|javascript)",
    r"exec\s*\(",
    r"eval\s*\(",
    r"__import__",
    r"os\.(system|popen|exec)",
]

# Compilar patrones para eficiencia
_COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) 
    for pattern in LLM_INJECTION_PATTERNS
]


def sanitize_text_for_llm(text: str) -> str:
    """Sanitize text before sending to LLM.
    
    Cleans text of potential injection attempts while
    preserving legitimate content.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text safe to send to LLM
    """
    if not text:
        return ""
    
    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Escape delimiters that could confuse the LLM
    sanitized = sanitized.replace("```", "'''")
    
    # Remove invisible unicode characters that could hide injections
    sanitized = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u206f]', '', sanitized)
    
    # Limit length to prevent context attacks
    max_length = 10000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated for length]"
    
    return sanitized.strip()


def sanitize_article_data(article: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize article data before using with LLM.
    
    Args:
        article: Dictionary with article data
        
    Returns:
        Article with sanitized text fields
    """
    sanitized = article.copy()
    
    text_fields = ['title', 'summary', 'content', 'description']
    for field in text_fields:
        if field in sanitized and sanitized[field]:
            sanitized[field] = sanitize_text_for_llm(str(sanitized[field]))
    
    return sanitized


def check_for_injection(text: str) -> bool:
    """Check if text contains injection patterns.
    
    Args:
        text: Text to check
        
    Returns:
        True if potential injection attempt detected
    """
    if not text:
        return False
    
    for pattern in _COMPILED_INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Potential LLM injection detected in input")
            return True
    
    return False


def safe_llm_input(text: str, field_name: str = "input") -> str:
    """Safely prepare text for sending to LLM.
    
    Combines sanitization and injection detection.
    If injection is detected, returns a safe placeholder.
    
    Args:
        text: Text to process
        field_name: Field name (for logging)
        
    Returns:
        Text safe for LLM
    """
    if check_for_injection(text):
        logger.warning(f"Injection blocked in {field_name}")
        return f"[Content sanitized - {field_name}]"
    
    return sanitize_text_for_llm(text)


def prepare_articles_for_llm(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare a list of articles for use with LLM.
    
    Args:
        articles: List of articles
        
    Returns:
        List of sanitized articles
    """
    return [sanitize_article_data(article) for article in articles]

# API Key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_from_settings() -> Optional[str]:
    """Get API key from settings (lazy import to avoid circular imports)."""
    try:
        from src.config import settings
        return getattr(settings, 'api_key', None) or None
    except Exception:
        return None


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """Verify the API key if authentication is enabled.
    
    Args:
        api_key: The API key from request header
        
    Returns:
        The verified API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    expected_key = get_api_key_from_settings()
    
    # If no API key configured, authentication is disabled
    if not expected_key:
        return None
    
    # If API key is configured, require valid key
    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Missing API key. Provide X-API-Key header."
        )
    
    if api_key != expected_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return api_key


# Dependency for protected routes
require_api_key = Depends(verify_api_key)
