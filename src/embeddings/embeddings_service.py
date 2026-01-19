"""Embeddings module for generating and comparing article embeddings."""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import AgglomerativeClustering

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Service for generating and comparing text embeddings using OpenAI."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        try:
            # Truncate text if too long (OpenAI has token limits)
            text = text[:8000]  # Roughly 8000 chars ~ 2000 tokens
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Truncate texts
                batch = [text[:8000] for text in batch]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                embeddings.extend([None] * len(batch))
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def cosine_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def is_duplicate(
        self, 
        embedding1: List[float], 
        embedding2: List[float],
        threshold: Optional[float] = None
    ) -> bool:
        """Check if two embeddings represent duplicate content."""
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
        
        Returns:
            List of duplicate groups, where each group is a list of indices.
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
        """Cluster articles by semantic similarity using embeddings.

        Args:
            articles: List of article dicts with 'embedding' key
            max_clusters: Maximum number of clusters to create
            distance_threshold: Distance threshold for clustering

        Returns:
            Dictionary mapping cluster_id to list of articles
        """
        if not articles:
            return {}

        # Extract embeddings
        embeddings = [a["embedding"] for a in articles if a.get("embedding")]
        if len(embeddings) < 2:
            return {0: articles}

        embeddings_array = np.array(embeddings)

        # Use Agglomerative Clustering - works well for semantic grouping
        n_clusters = min(max_clusters, len(articles))

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )

        labels = clustering.fit_predict(embeddings_array)

        # Group articles by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for article, label in zip(articles, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(article)

        logger.info(f"Created {len(clusters)} clusters from {len(articles)} articles")
        return clusters

    async def generate_cluster_name(
        self,
        articles: List[Dict[str, Any]]
    ) -> str:
        """Generate a descriptive name for a cluster based on article titles.

        Args:
            articles: List of articles in the cluster

        Returns:
            Short descriptive name for the cluster/topic
        """
        if not articles:
            return "General"

        titles = [a.get("title", "") for a in articles[:5]]  # Use up to 5 titles
        titles_text = "\n".join(f"- {t}" for t in titles if t)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that categorizes news articles. Given a list of article titles, respond with a short topic name (2-4 words max) that describes the common theme. Be specific and concise. Examples: 'AI & Machine Learning', 'Cloud Infrastructure', 'Startup Funding', 'Cybersecurity', 'Developer Tools'."
                    },
                    {
                        "role": "user",
                        "content": f"What topic best describes these articles?\n\n{titles_text}"
                    }
                ],
                max_tokens=20,
                temperature=0.3
            )

            name = response.choices[0].message.content.strip()
            logger.info(f"Generated cluster name: {name}")
            return name

        except Exception as e:
            logger.error(f"Error generating cluster name: {e}")
            return "General News"

    async def cluster_and_name_articles(
        self,
        articles: List[Dict[str, Any]],
        max_clusters: int = 8
    ) -> List[Dict[str, Any]]:
        """Cluster articles and generate names for each cluster.

        Args:
            articles: List of articles with embeddings
            max_clusters: Maximum number of clusters

        Returns:
            List of dicts with 'topic_name' and 'articles' keys, sorted by size
        """
        clusters = self.cluster_articles(articles, max_clusters)

        result = []
        for cluster_id, cluster_articles in clusters.items():
            topic_name = await self.generate_cluster_name(cluster_articles)
            result.append({
                "topic_name": topic_name,
                "articles": cluster_articles
            })

        # Sort by number of articles (largest first)
        result.sort(key=lambda x: len(x["articles"]), reverse=True)

        logger.info(f"Clustered {len(articles)} articles into {len(result)} topics")
        return result

    async def generate_executive_summary(
        self,
        grouped_articles: List[Dict[str, Any]]
    ) -> str:
        """Generate an executive summary of all news for a tech consultant.

        Args:
            grouped_articles: List of topic groups with articles

        Returns:
            Executive summary text (3-5 bullet points)
        """
        # Collect all titles grouped by topic
        summary_input = ""
        for group in grouped_articles:
            topic = group["topic_name"]
            titles = [a.get("title", "") for a in group["articles"][:3]]
            summary_input += f"\n{topic}:\n"
            summary_input += "\n".join(f"- {t}" for t in titles if t)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a tech industry analyst writing for freelance tech consultants.
Generate a brief executive summary (3-5 bullet points) of today's tech news.
Focus on:
- Actionable insights for consultants
- Emerging trends to watch
- Potential opportunities or risks for tech businesses
Keep it concise, professional, and forward-looking. No fluff."""
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these tech news topics for a consultant:\n{summary_input}"
                    }
                ],
                max_tokens=300,
                temperature=0.4
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Generated executive summary")
            return summary

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Executive summary unavailable."

    async def select_top_articles(
        self,
        articles: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Select the most important articles for a tech consultant.

        Args:
            articles: All processed articles
            top_n: Number of top articles to select

        Returns:
            List of top articles with 'relevance_reason' added
        """
        if len(articles) <= top_n:
            return articles

        # Prepare titles for ranking
        titles_text = "\n".join(
            f"{i+1}. {a.get('title', 'No title')} (Source: {a.get('source', 'Unknown')})"
            for i, a in enumerate(articles[:20])  # Limit to first 20 for token efficiency
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a tech industry analyst helping freelance tech consultants prioritize their reading.
Select the 3 most important articles based on:
1. Strategic importance (industry shifts, major releases, funding rounds)
2. Actionability (things a consultant should know to advise clients)
3. Timeliness (breaking news over evergreen content)

Respond in JSON format:
{"selections": [{"index": 1, "reason": "brief reason"}, {"index": 5, "reason": "brief reason"}, {"index": 3, "reason": "brief reason"}]}"""
                    },
                    {
                        "role": "user",
                        "content": f"Select the top 3 most important articles:\n\n{titles_text}"
                    }
                ],
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            selections = result.get("selections", [])

            top_articles = []
            for sel in selections[:top_n]:
                idx = sel.get("index", 1) - 1  # Convert to 0-indexed
                if 0 <= idx < len(articles):
                    article = articles[idx].copy()
                    article["relevance_reason"] = sel.get("reason", "")
                    top_articles.append(article)

            logger.info(f"Selected {len(top_articles)} top articles")
            return top_articles

        except Exception as e:
            logger.error(f"Error selecting top articles: {e}")
            # Fallback: return first N articles
            return articles[:top_n]

    async def generate_section_brief(
        self,
        topic_name: str,
        articles: List[Dict[str, Any]]
    ) -> str:
        """Generate a brief summary for a topic section.

        Args:
            topic_name: Name of the topic
            articles: Articles in this topic

        Returns:
            2-3 sentence brief about the topic
        """
        titles = [a.get("title", "") for a in articles[:5]]
        titles_text = "\n".join(f"- {t}" for t in titles if t)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a tech analyst writing section summaries for busy consultants.
Write a 2-3 sentence brief that:
- Highlights the key theme or trend
- Notes any actionable insights
- Is direct and professional, no fluff"""
                    },
                    {
                        "role": "user",
                        "content": f"Write a brief for the '{topic_name}' section with these articles:\n{titles_text}"
                    }
                ],
                max_tokens=100,
                temperature=0.4
            )

            brief = response.choices[0].message.content.strip()
            return brief

        except Exception as e:
            logger.error(f"Error generating section brief: {e}")
            return ""

    async def generate_section_narrative(
        self,
        topic_name: str,
        articles: List[Dict[str, Any]]
    ) -> str:
        """Generate a narrative that tells the story of what's happening in this topic.

        This creates podcast-ready content that connects articles into a cohesive narrative
        suitable for NotebookLM or audio generation.

        Args:
            topic_name: Name of the topic
            articles: Articles in this topic

        Returns:
            2-4 paragraph narrative telling the story of this topic
        """
        # Prepare article summaries for the narrative
        articles_input = ""
        for i, a in enumerate(articles[:6], 1):  # Limit to 6 articles
            title = a.get("title", "")
            content = a.get("content", "")[:400]  # First 400 chars of content
            source = a.get("source", "")
            articles_input += f"\n{i}. {title}\nSource: {source}\nSummary: {content}\n"

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a tech journalist writing a narrative summary for a podcast aimed at freelance tech consultants.

Write a 2-4 paragraph narrative that:
- Tells the STORY of what's happening in this topic area
- Connects the different news items into a cohesive narrative
- Uses a conversational but professional tone (like you're explaining to a colleague)
- Highlights what consultants should pay attention to and why
- Mentions specific companies, technologies, or trends by name
- Avoids bullet points - write in flowing paragraphs
- Sounds natural when read aloud

Think of it as: "Here's what's happening in [topic] this week and why it matters to your consulting practice..."

Do NOT just list the articles. Weave them into a story."""
                    },
                    {
                        "role": "user",
                        "content": f"Write a narrative for the '{topic_name}' section based on these articles:\n{articles_input}"
                    }
                ],
                max_tokens=500,
                temperature=0.6
            )

            narrative = response.choices[0].message.content.strip()
            logger.info(f"Generated narrative for {topic_name}")
            return narrative

        except Exception as e:
            logger.error(f"Error generating section narrative: {e}")
            return ""

    async def enrich_grouped_articles(
        self,
        grouped_articles: List[Dict[str, Any]],
        all_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enrich grouped articles with executive summary, top picks, briefs, and narratives.

        Args:
            grouped_articles: Articles grouped by topic
            all_articles: Flat list of all articles

        Returns:
            Dict with 'executive_summary', 'top_articles', and 'topics' (with briefs and narratives)
        """
        # Generate all enrichments in parallel where possible
        executive_summary = await self.generate_executive_summary(grouped_articles)
        top_articles = await self.select_top_articles(all_articles)

        # Generate briefs and narratives for each section
        enriched_topics = []
        for group in grouped_articles:
            brief = await self.generate_section_brief(
                group["topic_name"],
                group["articles"]
            )
            narrative = await self.generate_section_narrative(
                group["topic_name"],
                group["articles"]
            )
            enriched_topics.append({
                "topic_name": group["topic_name"],
                "brief": brief,
                "narrative": narrative,
                "articles": group["articles"]
            })

        logger.info("Enriched articles with summaries, briefs, and narratives")
        return {
            "executive_summary": executive_summary,
            "top_articles": top_articles,
            "topics": enriched_topics
        }


async def main():
    """Test embeddings service."""
    service = EmbeddingsService()
    
    texts = [
        "OpenAI releases GPT-5 with improved capabilities",
        "OpenAI announces GPT-5 with enhanced features",
        "Google unveils new AI model competing with GPT",
    ]
    
    embeddings = await service.generate_embeddings_batch(texts)
    
    print("\nSimilarity matrix:")
    for i, emb1 in enumerate(embeddings):
        for j, emb2 in enumerate(embeddings):
            if i < j and emb1 and emb2:
                sim = service.cosine_similarity(emb1, emb2)
                print(f"Text {i} vs Text {j}: {sim:.4f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
