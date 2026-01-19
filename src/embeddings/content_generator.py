"""Content generation module with LLM.

Contains logic for generating narrative content
for grouped articles using OpenAI.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from src.config import settings
from src.embeddings.prompts import (
    CLUSTER_NAME_SYSTEM,
    EXECUTIVE_SUMMARY_SYSTEM,
    SECTION_BRIEF_SYSTEM,
    SECTION_NARRATIVE_SYSTEM,
    TOP_ARTICLES_SYSTEM,
    LLM_CONFIG,
    cluster_name_prompt,
    executive_summary_prompt,
    section_brief_prompt,
    section_narrative_prompt,
    top_articles_prompt,
)
from src.security import (
    safe_llm_input,
    safe_log_error,
    sanitize_article_data,
    prepare_articles_for_llm,
)

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generates narrative content using LLM for grouped articles."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the content generator.
        
        Args:
            openai_api_key: OpenAI API key (optional, uses settings by default)
        """
        self.api_key = openai_api_key or settings.openai_api_key
        self.client = OpenAI(api_key=self.api_key)
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        config_key: str
    ) -> Optional[str]:
        """Call the LLM with specified configuration.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            config_key: Configuration key in LLM_CONFIG
            
        Returns:
            LLM response or None on error
        """
        config = LLM_CONFIG.get(config_key, {})
        
        try:
            kwargs = {
                "model": config.get("model", "gpt-4o-mini"),
                "max_tokens": config.get("max_tokens", 100),
                "temperature": config.get("temperature", 0.4),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Add response_format if configured
            if "response_format" in config:
                kwargs["response_format"] = config["response_format"]
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            safe_log_error(logger, f"LLM call failed ({config_key})", e)
            return None
    
    def generate_cluster_name(self, articles: List[Dict[str, Any]]) -> str:
        """Generate a descriptive name for a cluster of articles.
        
        Args:
            articles: List of articles in the cluster
            
        Returns:
            Cluster name (2-4 words)
        """
        if not articles:
            return "General News"
        
        # Sanitize and prepare titles
        titles = []
        for article in articles[:10]:  # Limitar para no exceder contexto
            safe_article = sanitize_article_data(article)
            title = safe_article.get('title', '')[:100]
            titles.append(f"- {title}")
        
        titles_text = "\n".join(titles)
        user_prompt = cluster_name_prompt(titles_text)
        
        result = self._call_llm(
            CLUSTER_NAME_SYSTEM,
            user_prompt,
            "cluster_name"
        )
        
        return result or "General News"
    
    def generate_executive_summary(
        self,
        grouped_articles: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[str]:
        """Generate executive summary of the day.
        
        Args:
            grouped_articles: Articles grouped by topic
            
        Returns:
            Executive summary in bullet point format
        """
        if not grouped_articles:
            return None
        
        # Preparar input con temas y conteos
        summary_parts = []
        for topic, articles in list(grouped_articles.items())[:10]:
            safe_topic = safe_llm_input(topic, "topic")
            sample_titles = []
            for article in articles[:3]:
                safe_article = sanitize_article_data(article)
                sample_titles.append(safe_article.get('title', '')[:80])
            
            summary_parts.append(
                f"**{safe_topic}** ({len(articles)} articles):\n" +
                "\n".join(f"  - {t}" for t in sample_titles)
            )
        
        summary_input = "\n\n".join(summary_parts)
        user_prompt = executive_summary_prompt(summary_input)
        
        return self._call_llm(
            EXECUTIVE_SUMMARY_SYSTEM,
            user_prompt,
            "executive_summary"
        )
    
    def select_top_articles(
        self,
        articles: List[Dict[str, Any]],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """Select the most important articles.
        
        Args:
            articles: List of candidate articles
            count: Number of articles to select
            
        Returns:
            List of selected articles with selection reason
        """
        if len(articles) <= count:
            return articles
        
        # Prepare numbered list of titles
        titles = []
        for i, article in enumerate(articles[:20], 1):
            safe_article = sanitize_article_data(article)
            title = safe_article.get('title', '')[:100]
            titles.append(f"{i}. {title}")
        
        titles_text = "\n".join(titles)
        user_prompt = top_articles_prompt(titles_text)
        
        result = self._call_llm(
            TOP_ARTICLES_SYSTEM,
            user_prompt,
            "top_articles"
        )
        
        if not result:
            return articles[:count]
        
        try:
            parsed = json.loads(result)
            selections = parsed.get("selections", [])
            
            selected = []
            for sel in selections[:count]:
                idx = sel.get("index", 1) - 1
                if 0 <= idx < len(articles):
                    article = articles[idx].copy()
                    article["selection_reason"] = sel.get("reason", "")
                    selected.append(article)
            
            return selected if selected else articles[:count]
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse top articles JSON response")
            return articles[:count]
    
    def generate_section_brief(
        self,
        topic: str,
        articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a short brief for a section.
        
        Args:
            topic: Topic/section name
            articles: Articles in the section
            
        Returns:
            Brief of 2-3 sentences
        """
        if not articles:
            return None
        
        safe_topic = safe_llm_input(topic, "topic")
        
        titles = []
        for article in articles[:5]:
            safe_article = sanitize_article_data(article)
            titles.append(f"- {safe_article.get('title', '')[:100]}")
        
        titles_text = "\n".join(titles)
        user_prompt = section_brief_prompt(safe_topic, titles_text)
        
        return self._call_llm(
            SECTION_BRIEF_SYSTEM,
            user_prompt,
            "section_brief"
        )
    
    def generate_section_narrative(
        self,
        topic: str,
        articles: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a podcast narrative for a section.
        
        Args:
            topic: Topic name
            articles: Articles in the section
            
        Returns:
            Narrative of 2-4 paragraphs
        """
        if not articles:
            return None
        
        safe_topic = safe_llm_input(topic, "topic")
        
        # Prepare articles with title and summary
        articles_parts = []
        for article in articles[:5]:
            safe_article = sanitize_article_data(article)
            title = safe_article.get('title', '')[:150]
            summary = safe_article.get('summary', '')[:300]
            articles_parts.append(f"**{title}**\n{summary}")
        
        articles_input = "\n\n".join(articles_parts)
        user_prompt = section_narrative_prompt(safe_topic, articles_input)
        
        return self._call_llm(
            SECTION_NARRATIVE_SYSTEM,
            user_prompt,
            "section_narrative"
        )
    
    def enrich_grouped_articles(
        self,
        grouped_articles: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Enrich grouped articles with generated content.
        
        Args:
            grouped_articles: Articles grouped by topic
            
        Returns:
            Dictionary with executive summary, top articles and enriched sections
        """
        result = {
            "executive_summary": None,
            "top_articles": [],
            "sections": {}
        }
        
        # Generate executive summary
        result["executive_summary"] = self.generate_executive_summary(grouped_articles)
        
        # Collect all articles for top selection
        all_articles = []
        for articles in grouped_articles.values():
            all_articles.extend(articles)
        
        # Select top articles
        result["top_articles"] = self.select_top_articles(all_articles)
        
        # Enrich each section
        for topic, articles in grouped_articles.items():
            result["sections"][topic] = {
                "articles": articles,
                "count": len(articles),
                "brief": self.generate_section_brief(topic, articles),
                "narrative": self.generate_section_narrative(topic, articles),
            }
        
        return result
